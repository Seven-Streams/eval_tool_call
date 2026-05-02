export SERVER_ADDR="127.0.0.1"
export SERVER_PORT="8000"
export MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct" # or the path of other model
export MODEL="Llama-3.1-8B-Instruct" # or other model names
export TOKENIZER="$MODEL_PATH" # or the path of other tokenizer
export DATA_PATH="./data/dataset"
export N_GPU=4
export ACC_RAW="./data/accuracy_raw"
export ACC_SUM="./data/accuracy_summary"
export EFF="./data/efficiency"



# Launch the server

# Launch SGLang server (1 GPU)
python -m sglang.launch_server --model-path $MODEL_PATH \
--host $SERVER_ADDR --port $SERVER_PORT --disable-radix-cache  --dtype float16 \
--enable-torch-compile 

# Launch SGLang server (multiple GPUs)
python -m sglang.launch_server --model-path $MODEL_PATH --tp 4 \
--host $SERVER_ADDR --port $SERVER_PORT --disable-radix-cache  --dtype float16 



# Test accuracy

python accuracy.py --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_parallel --num-requests 200 \
--dataset-path $DATA_PATH --num-gpus $N_GPU \
--num-warmup-requests 1 --request-rate inf \
--host $SERVER_ADDR --port $SERVER_PORT --api-endpoint sglang --output $ACC_RAW \
--temperature 0.001 --top-p 0.9 

python accuracy.py --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_parallel --num-requests 200 \
--dataset-path $DATA_PATH --num-gpus $N_GPU \
--num-warmup-requests 1 --request-rate inf \
--host $SERVER_ADDR --port $SERVER_PORT --api-endpoint sglang --output $ACC_RAW \
--temperature 0.001 --top-p 0.9 \
--use-stag

python accuracy.py --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_live_multiple --num-requests 1052 \
--dataset-path $DATA_PATH --num-gpus $N_GPU \
--num-warmup-requests 1 --request-rate inf \
--host $SERVER_ADDR --port $SERVER_PORT --api-endpoint sglang --output $ACC_RAW \
--temperature 0.001 --top-p 0.9 

python accuracy.py --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_live_multiple --num-requests 1052 \
--dataset-path $DATA_PATH --num-gpus $N_GPU \
--num-warmup-requests 1 --request-rate inf \
--host $SERVER_ADDR --port $SERVER_PORT --api-endpoint sglang --output $ACC_RAW \
--temperature 0.001 --top-p 0.9 \
--use-stag

# Process the the output
python check.py --dataset ALL --model ALL --dataset-path $DATA_PATH \
--output-root $ACC_RAW --final-root $ACC_SUM

python draw_accuracy.py --summary-root $ACC_SUM



# Test the efficiency

# With SGLang server
python efficiency.py  --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_multiple --dataset-path ./data/dataset --num-gpus $N_GPU \
--num-warmup-requests 200 --num-requests 200 \
--host $SERVER_ADDR --port $SERVER_PORT --num-concurrent-requests 1 \
--api-endpoint sglang --output $EFF --stream \
--temperature 0.001 --top-p 0.9 \
--use-stag

python efficiency.py  --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_multiple --dataset-path ./data/dataset --num-gpus $N_GPU \
--num-warmup-requests 200 --num-requests 200 \
--host $SERVER_ADDR --port $SERVER_PORT --num-concurrent-requests 1 \
--api-endpoint sglang --output $EFF --stream \
--temperature 0.001 --top-p 0.9 

# Process the the output
python check.py --dataset ALL --model ALL --dataset-path $DATA_PATH \
--output-root $ACC_RAW --final-root $ACC_SUM

python draw_efficiency.py --bench-root $ACC_SUM