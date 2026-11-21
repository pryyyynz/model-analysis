#!/usr/bin/env python3
import argparse
import json
import subprocess
import urllib.request

def nvidia():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    parts = [x.strip() for x in out.strip().split(",")]
    return {"vram_used_mb": float(parts[0]), "vram_total_mb": float(parts[1]), "gpu_util_pct": float(parts[2])}

def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read().decode()
    except Exception as e:
        return f"ERROR: {e}"

def parse_prom(text):
    vals = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if " " in line:
            key, val = line.rsplit(" ", 1)
            vals[key] = val
    return vals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--metrics-url", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = fetch(args.metrics_url)
    rec = {"engine": args.engine, "metrics_url": args.metrics_url, "nvidia": nvidia()}
    if not raw.startswith("ERROR"):
        parsed = parse_prom(raw)
        # keep a few useful keys if present
        interesting = [k for k in parsed if any(x in k for x in ("kv", "cache", "requests_running", "requests_waiting"))]
        rec["engine_metrics"] = {k: parsed[k] for k in interesting}
        rec["raw_metrics_excerpt"] = raw[:2000]
    else:
        rec["error"] = raw
    print(json.dumps(rec))
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
