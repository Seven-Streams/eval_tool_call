"""MLC LLM benchmark main entrance"""

import argparse
import functools
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from protocol.openai_api_protocol import ChatToolCall
import numpy as np
import requests
from transformers import AutoTokenizer  # pylint: disable=import-error

from api_endpoint import SUPPORTED_BACKENDS, create_api_endpoint
from dataset import SUPPORTED_DATASET, Dataset, create_dataset
from request_processor import (
    MetricAnalyzer,
    RequestProcessor,
    create_pipelines,
)
from request_record import (
    RequestRecord,
    generate_metrics_summary,
)



def _parse_num_concurrent_requests(num_str: Optional[str]) -> Optional[List[int]]:
    if num_str is None:
        return None
    numbers = num_str.split(",")
    if any(not number.isdigit() for number in numbers):
        raise ValueError(f"Unrecognized num_concurrent_requests list: {numbers}")
    return list(int(number) for number in numbers)


def _parse_request_rate(request_rate_str: Optional[str]) -> Optional[List[np.float32]]:
    if request_rate_str is None:
        return None
    request_rates = request_rate_str.split(",")
    results = []
    for rate_str in request_rates:
        request_rate = float(rate_str)
        if request_rate <= 0:
            raise ValueError(f"Invalid request rate {request_rate}")
        results.append(np.float32(request_rate))
    return results


def convert_calls_to_json(stringified_calls: str, model: str) -> List[Dict[str, Any]]:
    """Convert the list of ChatToolCall to a list of dict."""
    function_calls_json = []
    if "Llama-3" in model:
        start = 0
        while True:
            index = stringified_calls.find('{"name":', start)
            if index == -1:
                break
            try:
                decoder = json.JSONDecoder()
                result, end_index = decoder.raw_decode(stringified_calls, index)
            except:  # pylint: disable=bare-except
                start = index + 1
                continue
            start = end_index
            if (
                not isinstance(result, dict)
                or "name" not in result
                or "parameters" not in result
            ):
                continue
            function_calls_json.append(
                {
                    "function": {
                        "name": result["name"],
                        "arguments": result["parameters"],
                    }
                }
            )
    elif "Qwen" in model:
        start = 0
        while True:
            index = stringified_calls.find('<tool_call>\n{"name":', start)
            if index == -1:
                break
            try:
                decoder = json.JSONDecoder()
                result, end_index = decoder.raw_decode(
                    stringified_calls, index + len("<tool_call>\n")
                )
            except:  # pylint: disable=bare-except
                start = index + 1
                continue
            start = end_index
            if (
                not isinstance(result, dict)
                or "name" not in result
                or "arguments" not in result
            ):
                continue
            function_calls_json.append(
                {"function": {"name": result["name"], "arguments": result["arguments"]}}
            )
    elif "gpt-oss" in model:
        tool_extract_pattern = re.compile(
            r"to=([a-zA-Z_][a-zA-Z0-9_.-]*)\s*<\|constrain\|>json<\|message\|>(.*?)(?:<\|call\|>|$)",
            re.DOTALL,
        )
        all_matches = tool_extract_pattern.findall(stringified_calls)
        for full_function_name, json_content in all_matches:
            try:
                params = json.loads(json_content)
                parts = full_function_name.split('.', 1)
                function_name = parts[1]
                function_calls_json.append(
                    {
                        "function": {
                            "name": function_name,
                            "arguments": params,
                        }
                    }
                )
            except json.JSONDecodeError as e:
                continue 
    return function_calls_json

def run_pipeline(
    pipeline: RequestProcessor,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[RequestRecord]]:
    """Run the pipeline with the given dataset and args. Return the benchmark report dict."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    request_records = dataset.generate_request_records(
        args.input_len,
        args.output_len,
        args.input_len_std,
        args.output_len_std,
    )
    request_records = pipeline(request_records)
    num_total_requests = (
        args.num_requests
        if not args.per_gpu_workload
        else args.num_requests * args.num_gpus
    )
    assert len(request_records) == num_total_requests
    sorted_requests: List[RequestRecord] = [None] * num_total_requests
    for request_record in request_records:
        assert request_record.request_id is not None
        assert sorted_requests[request_record.request_id] is None
        sorted_requests[request_record.request_id] = request_record

    request_records = MetricAnalyzer(tokenizer)(request_records)
    report = generate_metrics_summary(
        request_records, num_total_requests, args.num_gpus
    )
    return report, sorted_requests


def query_mlc_server_metrics(host: str, port: int):
    """Try to get the MLC server metrics whenever it exists."""
    try:
        r = requests.post(
            f"http://{host}:{port}/debug/dump_engine_metrics", json={}, timeout=10
        )
        if r.status_code == 200:
            print(f"MLC server metrics: {r.json()}")
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def main(args: argparse.Namespace):
    """Main benchmark entrance."""
    if args.num_requests <= 0:
        raise ValueError("Number of requests to benchmark must be positive.")

    def _main():
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        dataset = create_dataset(args, tokenizer)
        dataset.require_fake_warmup = True
        f_create_api_endpoint = functools.partial(create_api_endpoint, args)
        pipelines = create_pipelines(args, f_create_api_endpoint, dataset)
        store_record = []
        output_dir = f"{args.output}/{args.model}/{args.dataset}/{'use_stag' if args.use_stag else 'no_stag'}/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i, pipeline in enumerate(pipelines):
            report, request_records = run_pipeline(pipeline, dataset, tokenizer, args)
            for request in request_records:
                store_record.append({"id": request.request_id})
                store_record[-1]["output"] = request.output_str
                store_record[-1]["call"] = convert_calls_to_json(
                    request.output_str, args.model
                )
            with open(f"{output_dir}/result.json", "w") as f:
                json.dump(store_record, f, indent=4)
    _main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLC LLM benchmark")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASET,
        help=f"The benchmark dataset kind. Supporting {SUPPORTED_DATASET}",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="The dataset file path.",
    )
    parser.add_argument(
        "--api-endpoint",
        type=str,
        choices=SUPPORTED_BACKENDS,
        default="openai",
        help="The API endpoint API for benchmarking.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="The path of the tokenizer directory.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="The number of GPUs used by the server. "
        "We need this to better analyze the throughput per GPU.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        required=True,
        help="The number of requests for benchmark.",
    )
    parser.add_argument(
        "--num-warmup-requests",
        type=int,
        help="The number of requests for warmup. "
        "It is optional when fixing the number of concurrent requests, and is required otherwise.",
    )
    parser.add_argument(
        "--per-gpu-workload",
        default=False,
        action="store_true",
        help='When set to True, the specified "num_concurrent_requests"/"request_rate" '
        "denote the workload **per GPU**, which means that the real values of "
        '"num_concurrent_requests"/"request_rate" used in benchmark'
        'will be multiplied by "num_gpus".',
    )
    parser.add_argument(
        "--num-concurrent-requests",
        type=_parse_num_concurrent_requests,
        help="The number(s) of concurrent requests to benchmark. "
        'It can be either one integer or a list of integer separated by commas(","). '
        "When specified, for each integer, the benchmark keeps these many consistent "
        "number of concurrently running requests.",
    )
    parser.add_argument(
        "--request-rate",
        type=_parse_request_rate,
        help="The request rate(s) denoting the number of new requests each second. "
        'It can be either one float number (or "inf") or a list of numbers separated '
        'by commas(","). '
        "When specified, the benchmark sends these many new requests each second. "
        'If it is "inf", all requests will be sent together at once.',
    )
    parser.add_argument(
        "--replay-timestamp-scale",
        type=float,
        help="The timestamp scale when replaying the timestamps in a dataset. "
        'The dataset replay mode is enabled when neither "--num-concurrent-requests" and '
        '"--request-rate" is specified. '
        "The scale is 1 by default in the replay mode.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        help="The benchmark request average input length. Default to None, "
        "which means the request input length depends on the dataset being used.",
    )
    parser.add_argument(
        "--input-len-std",
        type=float,
        default=0,
        help="The benchmark request input length standard deviation. Default to 0.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        help="The benchmark request average output length. Default to None, "
        "which means the request output length depends on the dataset being used.",
    )
    parser.add_argument(
        "--output-len-std",
        type=float,
        default=0,
        help="The benchmark request output length standard deviation. Default to 0.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Whether to benchmark stream responses. "
        "When not enabled, metrics such as time-to-first-token (TTFT) will not be available. "
        "Default to False.",
    )
    parser.add_argument(
        # NOTE: The current implementation of server metrics still has some issues that need fixes,
        # which makes it not work to include server metrics.
        "--include-server-metrics",
        action="store_true",
        help="Whether to also benchmark the server side request metrics. "
        "This option is only available when benchmarking MLC server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="The host address of the backend API.",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="The port of the backend API.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3 * 60 * 60,
        help="The timeout limit of each request.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random number seed. Default to 0.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature value for logit adjustment. Default to 1.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="The top-p value for sampling. Default to 1.",
    )
    parser.add_argument(
        "--ignore-eos",
        default=False,
        action="store_true",
        help='Whether to set the "ignore_eos" field.',
    )
    parser.add_argument(
        "--apply-chat-template",
        default=False,
        action="store_true",
        help="Whether to apply chat template to the request input text. "
        'It is not supported when "--input-len" is specified.',
    )
    parser.add_argument(
        "--num-process-workers",
        type=int,
        help="The number of parallel process workers to send the requests.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Whether to disable showing progress bar with tqdm during benchmarking.",
    )
    parser.add_argument(
        "--max-schedule-gap",
        type=float,
        default=0.5,
        help="The maximum allowed delay between the scheduled time in seconds.",
    )
    parser.add_argument(
        "--debug-dump",
        default=False,
        action="store_true",
        help="Whether to dump all request record raw data to file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="mlc_benchmark.csv",
        help="The path of the output file where to dump the benchmark results.",
    )
    parser.add_argument(
        "--use-stag",
        action="store_true",
        help="Whether to set structural tag.",
    )
    parser.add_argument(
        "--force-call",
        action="store_true",
        help="Whether to force call single tool.",
    )
    main(parser.parse_args())
    
    
