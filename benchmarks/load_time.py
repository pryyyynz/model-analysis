#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
import urllib.request

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", required=True, help="Command to start server")
    ap.add_argument("--health-url", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    proc = subprocess.Popen(args.cmd, shell=True)
    ready = False
    while time.time() - t0 < args.timeout:
        try:
            urllib.request.urlopen(args.health_url, timeout=5)
            ready = True
            break
        except Exception:
            time.sleep(2)
    elapsed = time.perf_counter() - t0
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    rec = {"load_time_s": elapsed if ready else None, "ready": ready, "command": args.cmd}
    print(json.dumps(rec))
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
