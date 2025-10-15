# Evaluate the tool-calling accuracy and efficiency on LLM engine with Structural Tag

The evaluation script is modified based on the MLC-LLM bench and BFCL ast checker. The script uses the Structural Tag API to test the tool-calling accuracy and efficiency

## Fast test for efficiency with SGLang backend

Run the script:
```bash
bash eff.sh
```

The bench data will be in `./src/data/efficiecy` directory. You can modify the parameters in the `eff.sh` file.


## Test the accuracy

First launch the server.
```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
--host 127.0.0.1 --port 30000 --disable-radix-cache \
--enable-torch-compile 
```

Than generate the raw data (w/ & w/o structural tag):
```bash
cd ./tool_call_eval
python accuracy.py --model Llama-3.1-8B-Instruct \
--tokenizer /dist/Llama-3-8B-Instruct \
--dataset BFCL_v3_simple --dataset-path ./data/dataset --num-gpus 1 \
--num-requests 400 --num-warmup-requests 1 --request-rate inf \
--host 127.0.0.1 --port 30000 \
--api-endpoint sglang --output ./data/accuracy_raw \
--temperature 0.001 --top-p 0.9 \
[--use-stag] [--force-call]
```

The raw data will be in `./src/data/accuracy_raw` directory. Finally process the raw data:
```bash
python check.py --dataset ALL --model ALL --dataset-path ./data/dataset \
--output-root ./data/accuracy_raw --final-root ./data/accuracy_summary
```
The detailed pictures will also be in `./src/data/accuracy_summary` directory. 

Note: you may need to modify `SUPPORTED_MODEL` and `SUPPORTED_DATASET` in `check.py`, as well as `models` and  `datasets` in the draw scripts accoring to the specific cases.

## Test the efficiency

Also first launch the server.
```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
--host 127.0.0.1 --port 30000 --disable-radix-cache  \
--enable-torch-compile 
```

Than generate the raw data (w/ & w/o structural tag, sglang backend):

```bash
python efficiency.py --model Llama-3.1-8B-Instruct \
--tokenizer /dist/Llama-3.1-8B-Instruct \
--dataset BFCL_v3_live_multiple --dataset-path ./data/dataset --num-gpus 1 \
--num-warmup-requests 1052 --num-requests 1052 \
--host 127.0.0.1 --port 30000 --num-concurrent-requests 128 \
--api-endpoint sglang --output ./data/efficiecy \
--temperature 0.001 --top-p 0.9 \
--stream [--use-stag]
```

The bench data will be in `./src/data/efficiecy` directory. 


Note: you may need to modify `models` and  `datasets`, as well as the desired metrics in `query_to_title` in the draw scripts accoring to the specific cases.