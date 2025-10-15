#!/bin/bash

# Define variables for easier modification
# =================================================
# NOTE: MODIFY THESE VARIABLES AS NEEDED
MODEL_PATH="/dist/Llama-3.2-1B-Instruct"
MODEL_NAME="Llama-3.2-1B-Instruct"
HOST="127.0.0.1"
PORT="30000"
DATASET="BFCL_v3_live_multiple"
DATASET_PATH="./data/dataset"
NUM_GPUS=1
NUM_REQUESTS=1052
BATCH_SIZE=128
OUTPUT_DIR="./data/efficiecy"
TEMP=0.001
TOP_P=0.9
API_ENDPOINT="sglang"
ENGINE_LAUNCH_TIME=60
# =================================================

# --- Step 1: Launch the SGLang server ---
echo "Starting the SGLang server..."
python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --disable-radix-cache \
  --enable-torch-compile &

# Get the process ID of the background server process
SERVER_PID=$!

# Wait for a moment to ensure the server has started
echo "Waiting for the server to initialize..."
sleep $ENGINE_LAUNCH_TIME

# --- Step 2: Generate raw data with and without structural tags ---
cd ./src
echo "Generating raw data without structural tags..."
python efficiency.py \
  --model "${MODEL_NAME}" \
  --tokenizer "${MODEL_PATH}" \
  --dataset "${DATASET}" \
  --dataset-path "${DATASET_PATH}" \
  --num-gpus "${NUM_GPUS}" \
  --num-warmup-requests "${NUM_REQUESTS}" \
  --num-requests "${NUM_REQUESTS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --num-concurrent-requests "${NUM_CONCURRENT_REQUESTS}" \
  --api-endpoint "${API_ENDPOINT}" \
  --output "${OUTPUT_DIR}" \
  --temperature "${TEMP}" \
  --top-p "${TOP_P}" \
  --stream

echo "Generating raw data with structural tags..."
python efficiency.py \
  --model "${MODEL_NAME}" \
  --tokenizer "${MODEL_PATH}" \
  --dataset "${DATASET}" \
  --dataset-path "${DATASET_PATH}" \
  --num-gpus "${NUM_GPUS}" \
  --num-warmup-requests "${NUM_REQUESTS}" \
  --num-requests "${NUM_REQUESTS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --num-concurrent-requests "${NUM_CONCURRENT_REQUESTS}" \
  --api-endpoint "${API_ENDPOINT}" \
  --output "${OUTPUT_DIR}" \
  --temperature "${TEMP}" \
  --top-p "${TOP_P}" \
  --stream \
  --use-stag

# --- Step 3: Kill the server process ---
echo "Stopping the SGLang server..."
kill "${SERVER_PID}"

echo "Script finished. Bench data is in the ${OUTPUT_DIR} directory."