"""
Usage:
# single GPU
python3 bench_speculative.py --model-path meta-llama/Llama-2-7b-chat-hf --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B

# multiple GPU
python3 bench_speculative.py --model-path deepseek-ai/DeepSeek-V3 --speculative-draft-model-path lmsys/DeepSeek-V3-NextN --tp-size 8 --trust-remote-code --batch-size 1 4 8 16 32 --steps 0 1 2 --topk 0 1 2 4 --num_draft_tokens 0 2 4 8
"""

import argparse
import asyncio
import datetime
import json
import os
import random
import time
from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict, Any, Literal

import numpy as np
import requests
import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sglang.bench_serving import (
    DatasetRow,
    set_global_args,
    is_file_valid_json,
    RequestFuncInput,
    async_request_profile,
    get_request,
    calculate_metrics,
    get_auth_headers,
    _get_bool_env_var,
    RequestFuncOutput,
    ASYNC_REQUEST_FUNCS
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)

async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[DatasetRow],
    request_rate: float,
    max_concurrency: Optional[int],
    disable_tqdm: bool,
    lora_names: List[str],
    extra_request_body: List[Dict[str, Any]],
    profile: bool,
    pd_separated: bool = False,
    flush_cache: bool = False,
    warmup_requests: int = 1,
    use_trace_timestamps: bool = False,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Limit concurrency
    # From https://github.com/vllm-project/vllm/pull/9390
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    # Warmup
    print(f"Starting warmup with {warmup_requests} sequences...")

    # For all other datasets, input_requests is a list of DatasetRow objects
    test_request = input_requests[0]

    if lora_names is not None and len(lora_names) != 0:
        lora_name = lora_names[0]
    else:
        lora_name = None

    # Create the test input once
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_request.prompt,
        api_url=api_url,
        prompt_len=test_request.prompt_len,
        output_len=min(test_request.output_len, 32),
        lora_name=lora_name,
        image_data=test_request.image_data,
        extra_request_body=extra_request_body[0],
    )

    # Run warmup requests
    warmup_tasks = []
    for _ in range(warmup_requests):
        warmup_tasks.append(
            asyncio.create_task(request_func(request_func_input=test_input))
        )

    warmup_outputs = await asyncio.gather(*warmup_tasks)

    # Check if at least one warmup request succeeded
    if warmup_requests > 0 and not any(output.success for output in warmup_outputs):
        raise ValueError(
            "Warmup failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {warmup_outputs[0].error}"
        )
    else:
        print(
            f"Warmup completed with {args.warmup_requests} sequences. Starting main benchmark run..."
        )

    # Flush cache
    if ("sglang" in backend and _get_bool_env_var("SGLANG_IS_IN_CI")) or flush_cache:
        requests.post(base_url + "/flush_cache", headers=get_auth_headers())

    time.sleep(1.0)

    # Start profiler
    if profile:
        print("Starting profiler...")
        profile_output = await async_request_profile(
            api_url=base_url + "/start_profile"
        )
        if profile_output.success:
            print("Profiler started")

    # Run all requests
    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    pbar_total = len(input_requests)
    request_generator = get_request(input_requests, request_rate)

    pbar = None if disable_tqdm else tqdm(total=pbar_total)
    async for i, request in enumerate(request_generator):
        if lora_names is not None and len(lora_names) != 0:
            idx = random.randint(0, len(lora_names) - 1)
            lora_name = lora_names[idx]
        else:
            lora_name = None

        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=request.output_len,
            lora_name=lora_name,
            image_data=request.image_data,
            extra_request_body=extra_request_body[i],
            timestamp=request.timestamp,
        )

        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    # Stop profiler
    if profile:
        print("Stopping profiler...")
        profile_output = await async_request_profile(api_url=base_url + "/stop_profile")
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    if "sglang" in backend:
        server_info = requests.get(
            base_url + "/get_server_info", headers=get_auth_headers()
        )
        if server_info.status_code == 200:
            server_info_json = server_info.json()
            if "decode" in server_info_json:
                server_info_json = server_info_json["decode"][0]
            if (
                "internal_states" in server_info_json
                and server_info_json["internal_states"]
            ):
                accept_length = server_info_json["internal_states"][0].get(
                    "avg_spec_accept_length", None
                )
            else:
                accept_length = None
        else:
            accept_length = None
    else:
        accept_length = None

    # Compute metrics and print results
    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics, output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        backend=backend,
    )

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print(
        "{:<40} {:<10}".format(
            "Traffic request rate:", "trace" if use_trace_timestamps else request_rate
        )
    )
    print(
        "{:<40} {:<10}".format(
            "Max request concurrency:",
            max_concurrency if max_concurrency else "not set",
        )
    )
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total input text tokens:", metrics.total_input_text))
    print(
        "{:<40} {:<10}".format("Total input vision tokens:", metrics.total_input_vision)
    )
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10}".format(
            "Total generated tokens (retokenized):", metrics.total_output_retokenized
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", metrics.total_throughput
        )
    )
    print("{:<40} {:<10.2f}".format("Concurrency:", metrics.concurrency))
    if accept_length:
        print("{:<40} {:<10.2f}".format("Accept length:", accept_length))
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median E2E Latency (ms):", metrics.median_e2e_latency_ms
        )
    )
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s="Inter-Token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P95 ITL (ms):", metrics.p95_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("{:<40} {:<10.2f}".format("Max ITL (ms):", metrics.max_itl_ms))
    print("=" * 50)

    if (
        metrics.median_ttft_ms is not None
        and metrics.mean_itl_ms is not None
        and metrics.output_throughput is not None
    ):
        result = {
            # Arguments
            "backend": args.backend,
            "dataset_name": args.dataset_name,
            "request_rate": "trace" if use_trace_timestamps else request_rate,
            "max_concurrency": max_concurrency,
            "sharegpt_output_len": args.sharegpt_output_len,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            # Results
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "total_input_text_tokens": metrics.total_input_text,
            "total_input_vision_tokens": metrics.total_input_vision,
            "total_output_tokens": metrics.total_output,
            "total_output_tokens_retokenized": metrics.total_output_retokenized,
            "request_throughput": metrics.request_throughput,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
            "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
            "std_e2e_latency_ms": metrics.std_e2e_latency_ms,
            "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
            "mean_ttft_ms": metrics.mean_ttft_ms,
            "median_ttft_ms": metrics.median_ttft_ms,
            "std_ttft_ms": metrics.std_ttft_ms,
            "p99_ttft_ms": metrics.p99_ttft_ms,
            "mean_tpot_ms": metrics.mean_tpot_ms,
            "median_tpot_ms": metrics.median_tpot_ms,
            "std_tpot_ms": metrics.std_tpot_ms,
            "p99_tpot_ms": metrics.p99_tpot_ms,
            "mean_itl_ms": metrics.mean_itl_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "std_itl_ms": metrics.std_itl_ms,
            "p95_itl_ms": metrics.p95_itl_ms,
            "p99_itl_ms": metrics.p99_itl_ms,
            "concurrency": metrics.concurrency,
            "accept_length": accept_length,
        }
    else:
        print(f"Error running benchmark for request rate: {request_rate}")
        print("-" * 30)

    # Determine output file name
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d")
        if args.dataset_name == "image":
            output_file_name = (
                f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_"
                f"{args.random_output_len}_{args.image_count}imgs_"
                f"{args.image_resolution}.jsonl"
            )
        elif args.dataset_name.startswith("random"):
            output_file_name = f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_{args.random_output_len}.jsonl"
        else:
            output_file_name = (
                f"{args.backend}_{now}_{args.num_prompts}_{args.dataset_name}.jsonl"
            )

    result_details = {
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    # Append results to a JSONL file
    with open(output_file_name, "a") as file:
        if args.output_details:
            result_for_dump = result | result_details
        else:
            result_for_dump = result
        file.write(json.dumps(result_for_dump) + "\n")

    return result | result_details


def generate_BFCL_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: int,
    num_requests: int = -1,
    apply_chat_template=False,
    stag_style: Literal["none", "Llama3", "Qwen3", "gpt-oss"] = "none",
) -> Tuple[List[DatasetRow], List[List[Dict]]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    if not is_file_valid_json(dataset_path) and dataset_path == "":
        raise ValueError("Please provide a valid dataset path.")
    with open(dataset_path) as f:
        dataset = json.load(f)

    if num_requests == -1:
        num_requests = len(dataset)
    if num_requests > len(dataset):
        raise ValueError(f"num_requests {num_requests} > dataset size {len(dataset)}")

    # Filter out sequences that are too long or too short
    output_dataset: List[DatasetRow] = []
    output_tools: List[List[Dict]] = []
    extra_bodies: List[Dict[str, Any]] = []
    for i in range(len(dataset)):
        if len(output_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        messages = dataset[i]["question"]
        tools = dataset[i]["tool"]

        if apply_chat_template:
            messages = tokenizer.apply_chat_template(
                conversation=messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
            if tokenizer.bos_token:
                messages = messages.replace(tokenizer.bos_token, "")

        prompt_len = len(tokenizer.encode(messages))
        output_len = fixed_output_len

        output_dataset.append(
            DatasetRow(
                prompt=messages,
                prompt_len=prompt_len,
                output_len=output_len,
            )
        )
        output_tools.append(tools)
        
        if apply_chat_template:
            if stag_style == "none":
                extra_bodies.append({})
            else:
                if stag_style == "Llama3":
                    response_format = {
                        "type": "structural_tag",
                        "format": {
                            "type": "triggered_tags",
                            "triggers": ["{\"name\":"],
                            "tags": [
                                {
                                    "begin": "{{\"name\": \"{func_name}\", \"parameters\": ".format(func_name=tool["function"]["name"]),
                                    "content": {
                                        "type": "json_schema", 
                                        "json_schema": {
                                                "properties": tool["function"]["parameters"]["properties"],
                                                "required": tool["function"]["parameters"]["required"],
                                                "type": tool["function"]["parameters"]["type"],
                                        },
                                    },
                                    "end": "}",
                                } for tool in tools
                            ],
                        },
                }
                elif stag_style == "Qwen3":
                    response_format = {
                        "type": "structural_tag",
                        "format": {
                            "type": "sequence",
                            "elements": [
                                {
                                    "type": "tag",
                                    "begin": "<think>\n",
                                    "content": {"type": "any_text"},
                                    "end": "\n</think>\n",
                                },
                                {
                                    "type": "triggered_tags",
                                    "triggers": ["\n<tool_call>"],
                                    "tags": [
                                        {
                                            "begin": '\n<tool_call>\n{{\"name\": \"{func_name}\", \"arguments\": '.format(func_name=tool["function"]["name"]),
                                            "content": {
                                                "type": "json_schema", 
                                                "json_schema": {
                                                        "properties": tool["function"]["parameters"]["properties"],
                                                        "required": tool["function"]["parameters"]["required"],
                                                        "type": tool["function"]["parameters"]["type"],
                                                },
                                            },
                                            "end": "}\n</tool_call>",
                                        } for tool in tools
                                    ],
                                }
                            ]
                        },
                    }
                elif stag_style == "gpt-oss":        
                    def from_builtin_tool_to_tag(tool) -> list[dict]:
                        tag = [
                            {
                                "begin": f"<|channel|>commentary to=functions.{tool["function"]["name"]} <|constrain|>json<|message|>",
                                "content": {
                                        "type": "json_schema", 
                                        "json_schema": {
                                            "properties": tool["function"]["parameters"]["properties"],
                                            "required": tool["function"]["parameters"]["required"],
                                            "type": tool["function"]["parameters"]["type"],
                                        },
                                },
                                "end": "<|call|>",
                            },
                            {
                                "begin": f"<|channel|>analysis to=functions.{tool["function"]["name"]} <|constrain|>json<|message|>",
                                "content": {
                                        "type": "json_schema", 
                                        "json_schema": {
                                            "properties": tool["function"]["parameters"]["properties"],
                                            "required": tool["function"]["parameters"]["required"],
                                            "type": tool["function"]["parameters"]["type"],
                                        },
                                },
                                "end": "<|call|>",
                            },
                        ]
                        return tag               
                    response_format = {
                        "type": "structural_tag",
                        "format": {
                            "type": "sequence",
                            "elements": [
                                {
                                    "type": "tag",
                                    "begin": "<|channel|>analysis<|message|>",
                                    "content": {"type": "any_text"},
                                    "end": "<|end|><|start|>assistant",
                                },
                                {
                                    "type": "triggered_tags",
                                    "tags": [],
                                    "triggers": ["<|channel|>analysis to=", "<|channel|>commentary to="],    
                                },
                            ]
                        },
                    }
                    for tool in tools:
                        response_format["format"]["elements"][1]["tags"].extend(from_builtin_tool_to_tag(tool))
                else:
                    raise ValueError(f"Unknown stag_style: {stag_style}")
                extra_bodies.append({"response_format": response_format})
        else:
            extra_bodies.append({
                "tools": tools,
                "tool_choice": "auto",
            })
        

    print(f"#Requests: {len(output_dataset)}")
    print(f"#Input tokens: {np.sum([x.prompt_len for x in output_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in output_dataset])}")
    return output_dataset, output_tools, extra_bodies


def node0_print(msg):
    if server_args.node_rank == 0:
        print(msg)

def send(base_url, num_requests, batch_size, tokenizer, dataset_path: str, apply_chat_template):
    if apply_chat_template:
        if "Llama3" in tokenizer.name_or_path:
            stag_style = "Llama3"
        elif "Qwen3" in tokenizer.name_or_path:
            stag_style = "Qwen3"
        elif "gpt-oss" in tokenizer.name_or_path:
            stag_style = "gpt-oss"
        else:
            raise ValueError(f"Please specify stag_style for model {tokenizer.name_or_path}")
    else:
        stag_style = "none"
    input_requests, tools, extra_bodies = generate_BFCL_dataset(
        dataset_path=dataset_path,
        num_requests=num_requests,
        tokenizer=tokenizer,
        fixed_output_len=1024,
        apply_chat_template=apply_chat_template,
        stag_style=stag_style,
    )
    backend = "sglang"
    api_url = f"{base_url}/generate" if apply_chat_template else f"{base_url}/v1/chat/completions"

    # We need to set some dummy values in order to call `benchmark` below.
    args = SimpleNamespace(
        disable_ignore_eos=False,
        disable_stream=False,
        return_logprob=False,
        backend=backend,
        dataset_name="custom",
        num_prompts=None,
        sharegpt_output_len=None,
        random_input_len=None,
        random_output_len=None,
        random_range_ratio=None,
        output_file=None,
        warmup_requests=1,
        output_details=False,
    )
    set_global_args(args)

    # Run benchmark
    results = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id="default",
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=float("inf"),
            max_concurrency=batch_size,
            disable_tqdm=False,
            lora_names=None,
            extra_request_body=extra_bodies,
            profile=None,
        )
    )

    assert results["completed"] == len(input_requests)
    return (results["mean_ttft_ms"], results["mean_tpot_ms"])


def main(args, server_args):
    base_url = "http://127.0.0.1:20000"

    for batch_size in args.batch_size:

        node0_print(
            f"Start Testing batch_size={batch_size} on dataset {args.dataset_path} with model {args.model_path}"
        )

        # Create an LLM.
        other_args = []
        other_args.extend(
            [
                "--cuda-graph-max-bs",
                batch_size,
                "--mem-fraction-static",
                server_args.mem_fraction_static,
                "--tp-size",
                server_args.tp_size,
                "--max-running-requests",
                batch_size,
            ]
        )
        if server_args.trust_remote_code:
            other_args.extend(
                [
                    "--trust-remote-code",
                ]
            )
        process = popen_launch_server(
            args.model_path,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={
                "SGLANG_RECORD_STEP_TIME": "1",
                **os.environ,
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=server_args.trust_remote_code
        )

        try:
            # Warmup
            _ = send(
                base_url=base_url,
                num_requests=-1,
                batch_size=batch_size,
                tokenizer=tokenizer,
                dataset_path=args.dataset_path,
                apply_chat_template=args.apply_chat_template
            )
            # Benchmark
            ttft_ms, tpot_ms = send(
                base_url=base_url,
                num_requests=-1,
                batch_size=batch_size,
                tokenizer=tokenizer,
                dataset_path=args.dataset_path,
                apply_chat_template=args.apply_chat_template
            )
        finally:
            kill_process_tree(process.pid)

        node0_print(
            f"batch_size={batch_size}, ttft_ms={ttft_ms}, tpot_ms={tpot_ms}"
        )

        record = {
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "batch_size": batch_size,
            "mean_ttft_ms": ttft_ms,
            "mean_tpot_ms": tpot_ms,
        }

        with open(args.output_path, "a") as fout:
            fout.write(json.dumps(record) + "\n")

        # Wait for the server to shutdown
        time.sleep(5)


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--batch-size", type=int, nargs="+", default=(1, 128))
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--apply-chat-template", action="store_true")
    args = parser.parse_args()
    server_args: ServerArgs = ServerArgs.from_cli_args(args)

    main(args, server_args)