# Clone Xgrammar 2 in devlopment.
mkdir llz
cd llz
git clone --recursive https://github.com/Seven-Streams/xgrammar.git
cd xgrammar
git pull --all
git switch main-dev/2025-10-22/jit_default

# Make sure the right version of g++.
conda install -c conda-forge gcc=12.1.0
conda install -c conda-forge gxx_linux-64

# Update the environment
uv sync
uv pip install ".[test]"
source ./.venv/bin/activate
cd ..

# Install sglang
git clone -b v0.5.3.post3 https://github.com/sgl-project/sglang.git
cd sglang
pip install --upgrade pip
pip install -e "python"
cd ..

# Make sure xgrammar is the right version
cd xgrammar
uv pip install ".[test]"
cd ..

# Clone eval_tool_call
git clone https://github.com/Seven-Streams/eval_tool_call.git
cd eval_tool_call
mkdir output

# Download the models
hf download Qwen/Qwen3-8B 
hf download meta-llama/Meta-Llama-3.1-8B-Instruct
hf download openai/gpt-oss-20b

# run the benchmark
python3 bench_stagv2.py --dataset-path ./dataset/BFCL_v3_live_multiple.json --batch-size 1 128  --model-path Qwen/Qwen3-8B --grammar-backend xgrammar --output-path ./output/qwen-xgr.json 
python3 bench_stagv2.py --dataset-path ./dataset/BFCL_v3_live_multiple.json --batch-size 1 128  --model-path Qwen/Qwen3-8B --grammar-backend llguidance --output-path ./output/qwen-llg.json 
python3 bench_stagv2.py --dataset-path ./dataset/BFCL_v3_live_multiple.json --batch-size 1 128  --model-path Qwen/Qwen3-8B --grammar-backend outlines --output-path ./output/qwen-outlines.json 
python3 bench_stagv2.py --dataset-path ./dataset/BFCL_v3_live_multiple.json --batch-size 1 128  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --grammar-backend xgrammar --output-path ./output/llama-xgr.json
python3 bench_stagv2.py --dataset-path ./dataset/BFCL_v3_live_multiple.json --batch-size 1 128  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --grammar-backend llguidance --output-path ./output/llama-llg.json
python3 bench_stagv2.py --dataset-path ./dataset/BFCL_v3_live_multiple.json --batch-size 1 128  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --grammar-backend outlines --output-path ./output/llama-outlines.json
python3 bench_stagv2.py --dataset-path ./dataset/BFCL_v3_live_multiple.json --batch-size 1 128  --model-path openai/gpt-oss-20b --grammar-backend xgrammar --output-path ./output/oss-xgr.json
python3 bench_stagv2.py --dataset-path ./dataset/BFCL_v3_live_multiple.json --batch-size 1 128  --model-path openai/gpt-oss-20b --grammar-backend llguidance --output-path ./output/oss-llg.json
python3 bench_stagv2.py --dataset-path ./dataset/BFCL_v3_live_multiple.json --batch-size 1 128  --model-path openai/gpt-oss-20b --grammar-backend outlines --output-path ./output/oss-outlines.json

# generate the results
cd ..
tar -cvzf results.tar.gz ./eval_tool_call/output
