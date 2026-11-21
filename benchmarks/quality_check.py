#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--keywords", default="", help="Comma-separated expected keywords")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompt = Path(args.prompt).read_text() if Path(args.prompt).exists() else args.prompt
    payload = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens, "temperature": 0.0, "stream": False}
    r = requests.post(args.endpoint, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    kws = [k.strip() for k in args.keywords.split(",") if k.strip()]
    hits = [k for k in kws if k.lower() in text.lower()]
    rec = {"output": text, "keywords": kws, "matched": hits, "all_matched": len(hits) == len(kws), "completion_tokens": data.get("usage", {}).get("completion_tokens")}
    print(json.dumps(rec))
    if args.out:
        Path(args.out).write_text(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
