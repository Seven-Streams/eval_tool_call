# Evaluate the tool-calling accuracy and efficiency on LLM engine with Structural Tag

The evaluation script is modified based on the MLC-LLM bench and BFCL ast checker. The script uses the Structural Tag API to test the tool-calling accuracy and efficiency

## Fast test for e2e efficiency with SGLang backend

Run the script:
```bash
python3 bench_e2e.py --dataset-path ./dataset/BFCL_v3_live_multiple.json --batch-size 1 128  --model-path /dist/Llama-3.2-1B-Instruct --output-path ./output/bench.json 
```

## Fast test for efficiency with SGLang backend with structural tag

Run the script:
```bash
python3 bench_stagv2.py --dataset-path ./dataset/BFCL_v3_live_multiple.json --batch-size 1 128  --model-path /dist/Llama-3.2-1B-Instruct --output-path ./output/bench.json 
```

