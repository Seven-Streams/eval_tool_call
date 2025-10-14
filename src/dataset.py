"""MLC LLM benchmark dataset classes"""

import argparse
import json
import os
import requests
import random
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd  # pylint: disable=import-error
from datasets import load_dataset  # pylint: disable=import-error
from transformers import AutoTokenizer  # pylint: disable=import-error

from mlc_llm.bench.request_record import GroupedRequestRecord, Metrics, RequestRecord
from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatToolCall,
    DebugConfig,
)


class Dataset:  # pylint: disable=too-few-public-methods
    """The dataset base class."""

    # We set a truncation limit of 100k.
    truncate_length = int(1e5)
    # For some that datasets (e.g., dataset that has shared common prefix),
    # we need fake warmup requests to avoid prefilling common prefixes to the engine.
    require_fake_warmup: bool = False
    # Whether the dataset contains timestamps already.
    # If the dataset comes with timestamps, the benchmark can just replay
    # the requests according to their timestamps.
    timestamp_available: bool = False

    def generate_request_records(
        self,
        input_len: Optional[int],
        output_len: Optional[int],
        input_len_std: float = 0.0,
        output_len_std: float = 0.0,
    ) -> List[RequestRecord]:
        """Get the raw unprocessed request records of the dataset."""
        raise NotImplementedError()


GORILLA_TO_OPENAPI = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}


from enum import IntEnum


class Err_type(IntEnum):
    FORMAT_ERROR = 0
    CALL_NUMBER_ERROR = 1
    FUNC_NAME_ERROR = 2
    PARA_KEY_ERROR = 3
    TYPE_ERROR = 4
    ENUM_ERROR = 5
    PARA_VALUE_ERROR = 6
    NONE = 7


class Error:
    def __init__(self, message: str = "", err_type: Err_type = Err_type.NONE):
        self.message = message
        self.error_type = err_type


class GorillaDataset(Dataset):  # pylint: disable=too-few-public-methods
    """The dataset class for Gorilla dataset.
    Reference: https://github.com/ShishirPatil/gorilla
    """

    def __init__(
        self,
        dataset: str,
        dataset_path: str,
        tokenizer: AutoTokenizer,
        use_stag: bool,
        force_call: bool,
        api_endpoint: str,
        model: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.use_stag = use_stag
        self.force_call = force_call
        self.api_endpoint = api_endpoint
        self.model = model
        self.gorilla_data = []
        base_url = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/data"
        id = 0
        dataset_file = f"{dataset_path}/{dataset}.json"
        if os.path.exists(dataset_file):
            with open(dataset_file, mode="r", encoding="utf-8") as file:
                self.gorilla_data = json.load(file)
        else:
            function_url = f"{base_url}/{dataset}.json"
            answer_url = f"{base_url}/possible_answer/{dataset}.json"
            print(f"Downloading {dataset}.json from GitHub...")
            functions_data = []
            answers_data = []
            try:
                function_response = requests.get(function_url)
                function_response.raise_for_status()
                function_text = function_response.text
                for line in function_text.strip().split("\n"):
                    if line.strip():
                        try:
                            functions_data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing function line in {dataset}.json: {e}")
                answer_response = requests.get(answer_url)
                answer_response.raise_for_status()
                answer_text = answer_response.text
                for line in answer_text.strip().split("\n"):
                    if line.strip():
                        try:
                            answers_data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing answer line in {dataset}.json: {e}")
                print(
                    f"Successfully downloaded {dataset}.json: {len(functions_data)} functions, {len(answers_data)} answers"
                )
            except requests.RequestException as e:
                print(f"Error downloading {dataset}.json: {e}")
                functions_data = []
                answers_data = []
            if not functions_data or not answers_data:
                print(f"Skipping {dataset}.json - failed to download data")
                return
            print(f"Processing {dataset}.json...")
            answers_by_id = {item["id"]: item for item in answers_data}
            for item in functions_data:
                item_id = item["id"]
                question = item["question"][0]
                if item_id not in answers_by_id:
                    print(f"Warning: No answer found for item {item_id}")
                    continue
                if "function" not in item or not item["function"]:
                    print(f"Warning: No function definition for item {item_id}")
                    continue
                tool = [
                    {"type": "function", "function": func} for func in item["function"]
                ]
                self.map_type_values(tool)
                answer = answers_by_id[item_id]
                if "ground_truth" not in answer or not answer["ground_truth"]:
                    print(f"Warning: No ground truth for item {item_id}")
                    continue
                ideal_call = []
                for ground_truth in answer["ground_truth"]:
                    function_name = list(ground_truth.keys())[0]
                    params = ground_truth[function_name]
                    ideal_call.append({"name": function_name, "arguments": params})
                self.gorilla_data.append(
                    {
                        "id": id,
                        "question": question,
                        "tool": tool,
                        "ideal_call": ideal_call,
                        "source": f"{dataset}.json",
                    }
                )
                id += 1
            with open(dataset_file, mode="w", encoding="utf-8") as file:
                json.dump(self.gorilla_data, file, ensure_ascii=False, indent=4)
        if self.tokenizer is not None:
            for item in self.gorilla_data:
                num_tokens = 0
                for message in item["question"]:
                    num_tokens += len(
                        tokenizer.encode(message["content"], add_special_tokens=False)
                    )
                item["num_tokens"] = num_tokens

    def gen_warmup_dataset(self):
        """Generate a warmup dataset for the benchmark."""
        length = len(self.gorilla_data)
        for i in range(length):
            self.gorilla_data.append(self.gorilla_data[i].copy())

    def generate_request_records(
        self,
        input_len: Optional[int],
        output_len: Optional[int],
        input_len_std: float = 0.0,
        output_len_std: float = 0.0,
    ) -> List[RequestRecord]:

        request_records = []
        for entry in self.gorilla_data:
            if output_len is not None:
                output_length = output_len
            else:
                output_length = 1024
            if self.use_stag:
                if "gpt-oss" in self.model:             
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
                            "type": "triggered_tags",
                            "tags": [],
                            "triggers": ["<|channel|>analysis to=", "<|channel|>commentary to="],
                            "stop_after_first": self.force_call,
                            # "at_least_one": self.force_call,
                        },
                    }
                    for tool in entry["tool"]:
                        response_format["format"]["tags"].extend(from_builtin_tool_to_tag(tool))
                elif "Qwen" in self.model:
                    response_format = {
                        "type": "structural_tag",
                        "format": {
                            "type": "triggered_tags",
                            "triggers": ["<tool_call>"],
                            "tags": [
                                {
                                    "begin": '<tool_call>\n{{\"name\": \"{func_name}\", \"arguments\": '.format(func_name=tool["function"]["name"]),
                                    "content": {
                                        "type": "json_schema", 
                                        "json_schema": {
                                                "properties": tool["function"]["parameters"]["properties"],
                                                "required": tool["function"]["parameters"]["required"],
                                                "type": tool["function"]["parameters"]["type"],
                                        },
                                    },
                                    "end": "}\n</tool_call>",
                                } for tool in entry["tool"]
                            ],
                            "stop_after_first": self.force_call,
                            # "at_least_one": self.force_call,
                        },
                    }
                elif "Llama" in self.model:
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
                                } for tool in entry["tool"]
                            ],
                            "stop_after_first": self.force_call,
                            "at_least_one": self.force_call,
                        },
                    }
                else:
                    raise NotImplementedError
            else:
                response_format = {
                    "type": "text",
                }
            
            if "Llama-3.1" in self.model:
                messages = [
                    ChatCompletionMessage(
                        content="",
                        role="system",
                    ),
                    ChatCompletionMessage(
                        content=(
                            "Given the following tool functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\n"
                            f"tools: {entry['tool']}\n\n"
                            'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.\n\n'
                        ),
                        role="user",
                    ),
                ]
            elif "Llama-3.2" in self.model:
                messages = [
                    ChatCompletionMessage(
                        content=(
                            "Tool Instructions:"
                            "You have access to the following tool functions:"
                            f"{entry['tool']}"
                            "If a you choose to call a function, you should ONLY reply in the following format:"
                            '`{"name": func_name, "parameters": parameters(JSON dict)}`'
                            "Here is an example,"
                            '`{"name": "get_time", "parameters": {"location": "Pittsburgh"}}}}`'
                            "Reminder:"
                            "- Function calls MUST follow the specified format"
                            "- Required parameters MUST be specified"
                            "- You should not repeat or miss the call"
                            "- You should response with at least one function calling"
                        ),
                        role="system",
                    )
                ]
            elif "Qwen" in self.model:
                tools_str = ""
                for tool in entry["tool"]:
                    tools_str += f"{json.dumps(tool, indent=4)}\n"
                messages = [
                    ChatCompletionMessage(
                        content=(
                            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
                            "# Tools\n\n"
                            "You may call one or more functions to assist with the user query.\n\n"
                            "You are provided with function signatures within <tools></tools> XML tags:\n"
                            f"<tools>\n{tools_str}</tools>\n\n"
                            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                            '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
                        ),
                        role="system",
                    ),
                    ChatCompletionMessage(content="", role="user"),
                ]
                for message in entry["question"]:
                    if message["role"] == "system":
                        messages[0].content += message["content"]
                    else:
                        messages[1].content += message["content"]
            elif "gpt-oss" in self.model:
                messages = [
                    ChatCompletionMessage(
                        content="",
                        role="system",
                    ),
                ]
            else:
                raise NotImplementedError
            for message in entry["question"]:
                if message["role"] == "system":
                    messages[0].content += message["content"]
                else:
                    messages.append(
                        ChatCompletionMessage(
                            content=message["content"], role=message["role"]
                        )
                    )
            request_records.append(
                RequestRecord(
                    request_id=entry["id"],
                    chat_cmpl=ChatCompletionRequest(
                        messages=messages,
                        response_format=response_format,
                        model="",
                        max_tokens=output_length,
                        tools=entry["tool"],
                        tool_choice="none" if ("Llama" in self.model or "Qwen" in self.model) else "auto",
                        debug_config=DebugConfig(grammar_execution_mode="constraint"),
                    ),
                    metrics=Metrics(
                        success=False,
                        start_time=0,
                        finish_time=0,
                        end_to_end_latency_s=0,
                        input_tokens=entry["num_tokens"],
                    ),
                )
            )
        return request_records


SUPPORTED_DATASET = [
    "BFCL_v3_simple",
    "BFCL_v3_multiple",
    "BFCL_v3_parallel",
    "BFCL_v3_live_simple",
    "BFCL_v3_live_multiple",
    "BFCL_v3_live_parallel",
]


def create_dataset(  # pylint: disable=too-many-return-statements,too-many-branches
    args: argparse.Namespace, tokenizer: AutoTokenizer
) -> Dataset:
    """Create a dataset instance with regard to the specified dataset kind and file path."""
    if args.dataset_path is not None and not isinstance(args.dataset_path, str):
        raise TypeError(
            f"Invalid dataset path {args.dataset_path}. Please use a string."
        )
    if args.dataset in SUPPORTED_DATASET:
        if args.dataset_path is None:
            raise ValueError(
                "Gorilla dataset requires dataset path. "
                'Please specify it with "--dataset-path".'
            )
        return GorillaDataset(
            args.dataset,
            args.dataset_path,
            tokenizer,
            args.use_stag,
            args.force_call,
            args.api_endpoint,
            args.model,
        )
    raise ValueError(f"Unrecognized dataset {args.dataset}")