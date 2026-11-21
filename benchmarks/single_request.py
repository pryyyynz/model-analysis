#!/usr/bin/env python3
import argparse
import json
import time
import uuid
from pathlib import Path

import requests

from metrics import GpuMemoryMonitor, system_ram_mb
from envinfo import base_metadata

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_once(endpoint, model, prompt, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    mon = GpuMemoryMonitor()
    mon.start()
    ram_before = system_ram_mb()
    ram_peak = ram_before
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    completion_tokens = None
    text = ""
    try:
        with requests.post(endpoint, json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    completion_tokens = obj["usage"].get("completion_tokens")
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}) or {}
                content = delta.get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
                    text += content
                ram = system_ram_mb()
                if ram > ram_peak:
                    ram_peak = ram
        total = time.perf_counter() - t0
    finally:
        mon.stop()
    if completion_tokens is None:
        completion_tokens = chunks
    tpot = None
    if completion_tokens and completion_tokens > 1 and ttft is not None:
        tpot = (total - ttft) / (completion_tokens - 1)
    toks_per_sec = completion_tokens / total if total > 0 else 0.0
    return {
        "ttft_s": ttft,
        "total_s": total,
        "completion_tokens": completion_tokens,
        "chunks": chunks,
        "tpot_s": tpot,
        "tokens_per_sec": toks_per_sec,
        "peak_vram_mb": mon.peak,
        "ram_before_mb": ram_before,
        "ram_peak_mb": ram_peak,
        "output_chars": len(text),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--experiment-id", default=None)
    args = ap.parse_args()

    prompt = Path(args.prompt).read_text() if Path(args.prompt).exists() else args.prompt
    meta = base_metadata(args.engine, args.model, args.quantization, args.model_revision, args.experiment_id)

    # warmup
    for _ in range(args.warmup):
        run_once(args.endpoint, args.model, prompt, args.max_tokens, args.timeout)

    results = []
    for i in range(args.runs):
        r = run_once(args.endpoint, args.model, prompt, args.max_tokens, args.timeout)
        r.update(meta)
        r.update({
            "engine": args.engine,
            "model": args.model,
            "run": i,
            "max_tokens": args.max_tokens,
            "prompt_chars": len(prompt),
            "id": str(uuid.uuid4()),
        })
        results.append(r)
        print(json.dumps(r))

    out_path = Path(args.out) if args.out else OUT_DIR / f"{args.engine}_{args.model.replace('/', '_')}.jsonl"
    with out_path.open("a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
