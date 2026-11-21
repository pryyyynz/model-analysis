# Concurrent LLM Inference on an 8 GB GPU

Companion repository for the Medium article [**"Fitting Is Not Serving: What an 8 GB GPU Can Actually Do With Concurrent LLM Requests"**](https://medium.com/@dugboryeleprince/fitting-is-not-serving-what-an-8-gb-gpu-can-actually-do-with-concurrent-llm-requests-4878f37632cb).

Controlled study of how quantization, context length, KV-cache allocation and inference-engine memory management determine the maximum sustainable concurrent workload on an NVIDIA RTX 5060 Laptop GPU (8151 MiB VRAM). Engines: vLLM, llama.cpp, Ollama.

## Layout

```
benchmarks/   load-generation harnesses (single request, context sweep, concurrency sweep)
analysis/     plot + summary generation (analyze.py, advanced_plots.py)
config/       sustainability criterion (p95 <= 2.0 s end-to-end, zero failures)
prompts/      fixed benchmark prompt
results/raw/  raw JSONL measurements (incl. excluded pilot records, kept for transparency)
results/plots/  6 main plots, 11 advanced plots, max_sustainable_concurrency.csv
```

Model weights (`models/`) and the llama.cpp source/build (`engines/`) are not tracked; see versions and acquisition below.

## Environment (pinned)

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB |
| Driver / CUDA | 595.99.02 / 13.2 (driver-reported) |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-43-generic, ext4 on NVMe |
| CPU / RAM | Intel Core 7 240H / 32 GB |
| Python | 3.12.16 (conda env `model_analysis`) |
| vLLM / torch | 0.33.0 / 2.14.0+cu130 |
| llama.cpp | build b12426, commit fc504f9, CUDA sm_120 |
| Ollama | 0.40.16 |
| pandas / matplotlib / seaborn | 3.1.1 / 3.11.2 / 0.13.2 |

## Models and server commands

Weights: Qwen2.5-0.5B/3B/7B-Instruct and Qwen2.5-7B-Instruct-AWQ from Hugging Face; Llama-3.1-8B-Instruct and Gemma-4-12B-it as GGUF (F16/Q8_0/Q4_K_M conversions via `llama-quantize`; only Q4_K_M and FP16/AWQ configurations were benchmarked). Ollama: library `qwen2.5:7b`, `llama3.1:8b`; local `gemma12b` Modelfile from the Gemma GGUF with `PARAMETER num_ctx 512`.

```bash
# vLLM (one at a time)
python -m vllm.entrypoints.openai.api_server --model <Qwen2.5-0.5B-Instruct> --port 8000 --max-model-len 8192
python -m vllm.entrypoints.openai.api_server --model <Qwen2.5-3B-Instruct> --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.95 --enforce-eager
python -m vllm.entrypoints.openai.api_server --model <Qwen2.5-7B-Instruct-AWQ> --port 8000 --quantization awq --max-model-len 2048 --gpu-memory-utilization 0.95

# llama.cpp (one at a time; this build defaults to 4 slots with a unified KV pool)
llama-server -m <Qwen2.5-7B-Instruct-Q4_K_M.gguf>   --port 8080 -ngl 99 -c 8192
llama-server -m <Llama-3.1-8B-Instruct-Q4_K_M.gguf> --port 8080 -ngl 99 -c 4096
llama-server -m <gemma-4-12b-it-Q4_K_M.gguf>        --port 8080 -ngl 99 -c 512

# Ollama (stock defaults; no OLLAMA_NUM_PARALLEL tuning)
ollama serve
```

## Reproducing the measurements

```bash
# per engine/model, against a running server:
python benchmarks/run_phase5.py --engine <engine> --model <model> --tokenizer <hf-id-or-path> \
  --endpoint http://127.0.0.1:<port>/v1/chat/completions --health-url http://127.0.0.1:<port>/health \
  --max-tokens 64 --context-runs 3 --concurrency-runs 3

# or individually:
python benchmarks/context_sweep.py     --engine ... --model ... --tokenizer ... --endpoint ... --lengths 512,1024,2048,4096,8192 --runs 3
python benchmarks/concurrency_sweep.py --engine ... --model ... --endpoint ... --prompt prompts/short.txt --levels 1,2,4,8,16 --runs 3

# analysis (filters pilot/setup records: pre-metadata, 32-token pilot, connection-refused):
python analysis/analyze.py
python analysis/advanced_plots.py
```

Workload: fixed 70-char prompt, 64-token output cap, `temperature=0`, streaming; concurrency levels 1/2/4/8/16 (12B: 1/2/4, 2 runs). Sustainability criterion fixed before testing: `config/sustainability.json` (p95 end-to-end <= 2.0 s, zero failed requests).

## Data notes

- `results/raw/` keeps all records, including excluded pilots (a 32-token pilot run, pre-metadata records, and connection-refused attempts against a server that was not up). Analysis scripts filter these; raw data is unmodified.
- Gemma-4-12B runs have no TTFT/TPOT: the harness captured usage totals but no content deltas for that model. Latency/throughput metrics only.
- Known failures: vLLM bitsandbytes 8-bit unsupported in 0.33.0; vLLM AWQ 7B KV-cache OOM at `--max-model-len 4096` (~0.16 GiB free vs ~0.22 GiB needed); vLLM 3B HTTP 400 for 8192-token prompts (`--max-model-len 4096`).
- Raw records contain local absolute paths from the experiment machine.