# Evaluate tool-calling accuracy and efficiency on SGLang with Structural Tag

The evaluation script is modified based on the BFCL ast checker. The script uses the Structural Tag API to test tool-calling accuracy and efficiency against an SGLang OpenAI-compatible server.

## Test the accuracy

First launch the server.
```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
--host 127.0.0.1 --port 8000 --dtype float16
```

Than generate the raw data (w/ & w/o structural tag):
```bash
cd ./tool_call_eval
python accuracy.py --model Llama-3.1-8B-Instruct \
--tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
--dataset BFCL_v3_simple --dataset-path ./data/dataset --num-gpus 1 \
--num-requests 400 --num-warmup-requests 1 --request-rate inf \
--host 127.0.0.1 --port 8000 \
--api-endpoint sglang --output ./data/accuracy_raw \
--temperature 0.001 --top-p 0.9 \
[--use-stag]
```

The raw data will be in `./data/accuracy_raw` directory. Finally process the raw data:
```bash
python check.py --dataset ALL --model ALL --dataset-path ./data/dataset \
--output-root ./data/accuracy_raw --final-root ./data/accuracy_summary
```

The summary wii be in `./data/accuracy_summary` directory. Run the script to generate summary pictures:
```bash
python draw_accuracy.py --summary-root ./data/accuracy_summary
```

The pictures will be in `./data/accuracy_summary` directory. To get more detailed picture, please run
```bash
python ./draw_accuracy_detail.py --summary-root ./data/accuracy_summary
```

The detailed pictures will also be in `./data/accuracy_summary` directory. 

Note: you may need to modify `SUPPORTED_MODEL` and `SUPPORTED_DATASET` in `check.py`, as well as `models` and  `datasets` in the draw scripts accoring to the specific cases.

## Test the efficiency

Also first launch the server.
```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
--host 127.0.0.1 --port 8000 --disable-radix-cache  --dtype float16 \
--enable-torch-compile 
```

Than generate the raw data (w/ & w/o structural tag):

```bash
python efficiency.py --model Llama-3.1-8B-Instruct \
--tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
--dataset BFCL_v3_multiple --dataset-path ./data/dataset --num-gpus 1 \
--num-warmup-requests 200 --num-requests 200 \
--host 127.0.0.1 --port 8000 --num-concurrent-requests 1 \
--api-endpoint sglang --output ./data/efficiecy \
--temperature 0.001 --top-p 0.9 \
--stream [--use-stag]
```

The bench data will be in `./data/efficiecy` directory. Run the script to generate summary pictures:
```bash
python draw_efficiency.py --bench-root ./data/efficiency
```

The pictures will be in `./data/efficiency` directory. 

Note: you may need to modify `models` and  `datasets`, as well as the desired metrics in `query_to_title` in the draw scripts accoring to the specific cases.