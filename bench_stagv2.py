import argparse
import json
import os
import time
from transformers import AutoTokenizer
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)
from support import send


def node0_print(msg):
    if server_args.node_rank == 0:
        print(msg)


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
                apply_chat_template=True,
                use_stag=True,
            )
            # Benchmark
            stag_ttft_ms, stag_tpot_ms, stag_generated_texts = send(
                base_url=base_url,
                num_requests=-1,
                batch_size=batch_size,
                tokenizer=tokenizer,
                dataset_path=args.dataset_path,
                apply_chat_template=True,
                use_stag=True
            )
            # Warmup
            _ = send(
                base_url=base_url,
                num_requests=-1,
                batch_size=batch_size,
                tokenizer=tokenizer,
                dataset_path=args.dataset_path,
                apply_chat_template=True,
                use_stag=False
            )
            # Benchmark
            no_stag_ttft_ms, no_stag_tpot_ms, no_stag_generated_texts = send(
                base_url=base_url,
                num_requests=-1,
                batch_size=batch_size,
                tokenizer=tokenizer,
                dataset_path=args.dataset_path,
                apply_chat_template=True,
                use_stag=False
            )
        finally:
            kill_process_tree(process.pid)

        node0_print(
            f"batch_size={batch_size}, with stag: ttft_ms={stag_ttft_ms}, tpot_ms={stag_tpot_ms}, without stag: ttft_ms={no_stag_ttft_ms}, tpot_ms={no_stag_tpot_ms}"
        )

        record = {
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "batch_size": batch_size,
            "with_stag": {
                "mean_ttft_ms": stag_ttft_ms,
                "mean_tpot_ms": stag_tpot_ms,
                "output": stag_generated_texts,
            },
            "without_stag": {
                "mean_ttft_ms": no_stag_ttft_ms,
                "mean_tpot_ms": no_stag_tpot_ms,
                "output": no_stag_generated_texts,
            },
        }
        if os.path.exists(args.output_path) and os.path.getsize(args.output_path) > 0:
            with open(args.output_path, "r") as file:
                all_records = json.load(file)
        else:
            all_records = {}
        if args.model_path not in all_records:
            all_records[args.model_path] = {}
        all_records[args.model_path][f"batch_size={batch_size}"] = record
        with open(args.output_path, "w") as file:
            json.dump(all_records, file, indent=4)

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
    args = parser.parse_args()
    server_args: ServerArgs = ServerArgs.from_cli_args(args)

    main(args, server_args)