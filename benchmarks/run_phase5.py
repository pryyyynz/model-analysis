#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / "benchmarks" / "single_request.py")

def wait_for(url, timeout=300):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except Exception:
            time.sleep(2)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--health-url", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--context-runs", type=int, default=3)
    ap.add_argument("--concurrency-runs", type=int, default=1)
    args = ap.parse_args()

    if not wait_for(args.health_url):
        raise SystemExit("server not ready")

    ctx_cmd = [
        "python", str(ROOT / "benchmarks" / "context_sweep.py"),
        "--engine", args.engine,
        "--model", args.model,
        "--tokenizer", args.tokenizer,
        "--endpoint", args.endpoint,
        "--max-tokens", str(args.max_tokens),
        "--runs", str(args.context_runs),
    ]
    conc_cmd = [
        "python", str(ROOT / "benchmarks" / "concurrency_sweep.py"),
        "--engine", args.engine,
        "--model", args.model,
        "--endpoint", args.endpoint,
        "--prompt", str(ROOT / "prompts" / "short.txt"),
        "--max-tokens", str(args.max_tokens),
        "--runs", str(args.concurrency_runs),
    ]
    print("Running context sweep...")
    subprocess.run(ctx_cmd, check=True)
    print("Running concurrency sweep...")
    subprocess.run(conc_cmd, check=True)
    print("Phase 5 run complete for", args.engine, args.model)

if __name__ == "__main__":
    main()
