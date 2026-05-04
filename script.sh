export SERVER_ADDR="127.0.0.1"
export SERVER_PORT="8000"
export MODEL_PATH="Qwen/Qwen3.6-27B" # or the path of other model
export MODEL="Qwen3.6-27B" # or other model names
export TOKENIZER="$MODEL_PATH" # or the path of other tokenizer
export DATA_PATH="./data/dataset"
export ACC_RAW="./data/accuracy_raw"
export ACC_SUM="./data/accuracy_summary"
export DATASET="BFCL_v3_simple"
export REQUEST_NUM=100

python -m sglang.launch_server --model-path $MODEL_PATH \
--host $SERVER_ADDR --port $SERVER_PORT --disable-radix-cache  --dtype float16 \
--enable-torch-compile 

python accuracy.py --model $MODEL --tokenizer $TOKENIZER \
--dataset $DATASET --num-requests $REQUEST_NUM \
--dataset-path $DATA_PATH --num-gpus $N_GPU \
--num-warmup-requests 1 --request-rate inf \
--host $SERVER_ADDR --port $SERVER_PORT --api-endpoint sglang --output $ACC_RAW \
--temperature 0.001 --top-p 0.9 

python accuracy.py --model $MODEL --tokenizer $TOKENIZER \
--dataset $DATASET --num-requests $REQUEST_NUM \
--dataset-path $DATA_PATH --num-gpus $N_GPU \
--num-warmup-requests 1 --request-rate inf \
--host $SERVER_ADDR --port $SERVER_PORT --api-endpoint sglang --output $ACC_RAW \
--temperature 0.001 --top-p 0.9 \
--use-stag

python check.py --dataset ALL --model ALL --dataset-path $DATA_PATH \
--output-root $ACC_RAW --final-root $ACC_SUM
