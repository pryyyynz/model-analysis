#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from single_request import run_once
from envinfo import base_metadata
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Memory bandwidth and capacity shape large language model serving. "
)


def make_prompt(tokenizer, target_tokens):
    toks = tokenizer.encode(BASE_TEXT, add_special_tokens=False)
    if not toks:
        return BASE_TEXT
    reps = (target_tokens // len(toks)) + 2
    text = BASE_TEXT * reps
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--experiment-id", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    lengths = [int(x) for x in args.lengths.split(",")]
    meta = base_metadata(args.engine, args.model, args.quantization, args.model_revision, args.experiment_id)

    wp = make_prompt(tok, lengths[0])
    for _ in range(args.warmup):
        try:
            run_once(args.endpoint, args.model, wp, args.max_tokens, args.timeout)
        except Exception:
            pass

    out_path = Path(args.out) if args.out else OUT_DIR / f"context_{args.engine}_{args.model.replace('/', '_')}.jsonl"
    with out_path.open("a") as f:
        for L in lengths:
            prompt = make_prompt(tok, L)
            actual = len(tok.encode(prompt, add_special_tokens=False))
            for i in range(args.runs):
                try:
                    r = run_once(args.endpoint, args.model, prompt, args.max_tokens, args.timeout)
                    r.update(meta)
                    r.update({
                        "engine": args.engine,
                        "model": args.model,
                        "target_context_tokens": L,
                        "actual_prompt_tokens": actual,
                        "run": i,
                        "max_tokens": args.max_tokens,
                    })
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    print(json.dumps(r))
                except Exception as e:
                    rec = {}
                    rec.update(meta)
                    rec.update({
                        "engine": args.engine,
                        "model": args.model,
                        "target_context_tokens": L,
                        "actual_prompt_tokens": actual,
                        "run": i,
                        "error": str(e),
                        "status": "failed",
                    })
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(json.dumps(rec))


if __name__ == "__main__":
    main()
