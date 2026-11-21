#!/usr/bin/env python3
import argparse
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from metrics import GpuMemoryMonitor, system_ram_mb
from envinfo import base_metadata

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def one_request(endpoint, model, prompt, max_tokens, timeout, idx):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    completion_tokens = None
    text = ""
    error = None
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
        total = time.perf_counter() - t0
    except Exception as e:
        total = time.perf_counter() - t0
        error = str(e)
        ttft = ttft
    if completion_tokens is None:
        completion_tokens = chunks
    tpot = None
    if completion_tokens and completion_tokens > 1 and ttft is not None:
        tpot = (total - ttft) / (completion_tokens - 1)
    return {
        "request_idx": idx,
        "ttft_s": ttft,
        "total_s": total,
        "completion_tokens": completion_tokens,
        "chunks": chunks,
        "tpot_s": tpot,
        "tokens_per_sec": (completion_tokens / total) if total > 0 else 0.0,
        "output_chars": len(text),
        "error": error,
        "ok": error is None,
    }


def run_level(endpoint, model, prompt, max_tokens, concurrency, timeout):
    mon = GpuMemoryMonitor()
    mon.start()
    ram_before = system_ram_mb()
    ram_peak = ram_before
    results = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one_request, endpoint, model, prompt, max_tokens, timeout, i) for i in range(concurrency)]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            ram = system_ram_mb()
            if ram > ram_peak:
                ram_peak = ram
    wall = time.perf_counter() - t_start
    mon.stop()
    ok = [r for r in results if r["ok"]]
    total_tokens = sum(r["completion_tokens"] or 0 for r in ok)
    agg_tps = total_tokens / wall if wall > 0 else 0.0
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    totals = [r["total_s"] for r in ok]
    summary = {
        "concurrency": concurrency,
        "n_requests": concurrency,
        "n_ok": len(ok),
        "n_failed": concurrency - len(ok),
        "wall_s": wall,
        "aggregate_tokens_per_sec": agg_tps,
        "total_tokens": total_tokens,
        "peak_vram_mb": mon.peak,
        "ram_before_mb": ram_before,
        "ram_peak_mb": ram_peak,
        "ttft_p50_s": sorted(ttfts)[len(ttfts)//2] if ttfts else None,
        "ttft_p95_s": sorted(ttfts)[int(0.95*(len(ttfts)-1))] if ttfts else None,
        "latency_p50_s": sorted(totals)[len(totals)//2] if totals else None,
        "latency_p95_s": sorted(totals)[int(0.95*(len(totals)-1))] if totals else None,
    }
    return summary, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--runs", type=int, default=1, help="repeat each concurrency level")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--experiment-id", default=None)
    args = ap.parse_args()

    prompt = Path(args.prompt).read_text() if Path(args.prompt).exists() else args.prompt
    levels = [int(x) for x in args.levels.split(",")]
    meta = base_metadata(args.engine, args.model, args.quantization, args.model_revision, args.experiment_id)

    # warmup
    one_request(args.endpoint, args.model, prompt, args.max_tokens, args.timeout, 0)

    out_path = Path(args.out) if args.out else OUT_DIR / f"concurrency_{args.engine}_{args.model.replace('/', '_')}.jsonl"
    with out_path.open("a") as f:
        for level in levels:
            for run in range(args.runs):
                summary, results = run_level(args.endpoint, args.model, prompt, args.max_tokens, level, args.timeout)
                summary.update(meta)
                summary.update({
                    "engine": args.engine,
                    "model": args.model,
                    "run": run,
                    "max_tokens": args.max_tokens,
                    "prompt_chars": len(prompt),
                    "id": str(uuid.uuid4()),
                })
                f.write(json.dumps({"type": "summary", **summary}) + "\n")
                for r in results:
                    r.update(meta)
                    r.update({"engine": args.engine, "model": args.model, "concurrency": level, "run": run})
                    f.write(json.dumps({"type": "request", **r}) + "\n")
                f.flush()
                print(json.dumps(summary))


if __name__ == "__main__":
    main()
