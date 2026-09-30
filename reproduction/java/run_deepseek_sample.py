#!/usr/bin/env python3
"""Run a small, resumable DeepSeek generation sample over prepared RAG inputs.

The API key is read only from DEEPSEEK_API_KEY and is never serialized.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:[A-Za-z0-9_+.#-]+)?\s*\n?(.*?)```", text, flags=re.DOTALL)
    return (blocks[-1] if blocks else text).strip()


def normalized(text: str) -> str:
    return text.replace(" _newline_ ", "\n").replace("_newline_", "\n").strip()


def request_completion(
    *, api_key: str, base_url: str, model: str, prompt: str, temperature: float, max_tokens: int,
    thinking: str
) -> tuple[dict, float]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior Java developer experienced in resolving merge conflicts. "
                    "Return only the complete resolved code, preferably in one fenced code block."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": thinking},
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result, time.monotonic() - started


def call_with_retry(**kwargs) -> tuple[dict, float]:
    for attempt in range(1, 4):
        try:
            return request_completion(**kwargs)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == 3:
                raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt == 3:
                raise RuntimeError(f"DeepSeek request failed after 3 attempts: {exc}") from exc
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--temperature", type=float, default=1.99)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    source = load_jsonl(args.input)[: args.sample_size]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if args.output.exists():
        completed = {x["global_id"]: x for x in load_jsonl(args.output)}
        # Results written by the first smoke run predate the explicit switch;
        # DeepSeek documents thinking=enabled as the API default.
        for row in completed.values():
            row.setdefault("thinking", "enabled")
            row.setdefault("max_tokens_requested", 4096)
            row.setdefault("temperature", 1.99)
            row.setdefault(
                "generation_complete",
                row.get("finish_reason") == "stop" and bool(row.get("extracted_resolution", "").strip()),
            )

    for index, item in enumerate(source, 1):
        gid = item["global_id"]
        if gid in completed and completed[gid].get("status") == "ok" and completed[gid].get("generation_complete"):
            print(f"[{index}/{len(source)}] id={gid} already complete", flush=True)
            continue
        try:
            response, latency = call_with_retry(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                prompt=item["prompt_for_llm"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                thinking=args.thinking,
            )
            raw = response["choices"][0]["message"]["content"]
            code = extract_code(raw)
            finish_reason = response["choices"][0].get("finish_reason")
            generation_complete = finish_reason == "stop" and bool(code.strip())
            record = {
                "global_id": gid,
                "repo": item["repo"],
                "status": "ok" if generation_complete else "incomplete",
                "model_requested": args.model,
                "model_returned": response.get("model"),
                "thinking": args.thinking,
                "max_tokens_requested": args.max_tokens,
                "temperature": args.temperature,
                "created": response.get("created"),
                "latency_seconds": round(latency, 3),
                "usage": response.get("usage", {}),
                "finish_reason": finish_reason,
                "generation_complete": generation_complete,
                "raw_response": raw,
                "extracted_resolution": code,
                "gold_resolution": item["gold_resolution"],
                "exact_match": bool(code.strip()) and normalized(code) == normalized(item["gold_resolution"]),
                "retrieved_global_ids": item["retrieved_global_ids"],
            }
        except Exception as exc:
            record = {
                "global_id": gid,
                "repo": item["repo"],
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        completed[gid] = record
        args.output.write_text(
            "".join(json.dumps(completed[x["global_id"]], ensure_ascii=False) + "\n" for x in source if x["global_id"] in completed),
            encoding="utf-8",
        )
        print(
            f"[{index}/{len(source)}] id={gid} status={record['status']} "
            f"latency={record.get('latency_seconds', '-')} exact={record.get('exact_match', '-')}",
            flush=True,
        )
        if record["status"] == "error":
            raise RuntimeError(record["error"])

    rows = [completed[x["global_id"]] for x in source if x["global_id"] in completed]
    api_ok = [x for x in rows if x["status"] in {"ok", "incomplete"}]
    complete = [x for x in rows if x.get("generation_complete")]
    incomplete = [x for x in rows if x["status"] == "incomplete"]
    errors = [x for x in rows if x["status"] == "error"]
    usage = defaultdict(int)
    for row in api_ok:
        for key, value in row.get("usage", {}).items():
            if isinstance(value, int):
                usage[key] += value
    summary = {
        "finished_at": datetime.now().astimezone().isoformat(),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "requested_samples": args.sample_size,
        "completed": len(rows),
        "api_successful": len(api_ok),
        "generation_complete": len(complete),
        "generation_incomplete": len(incomplete),
        "errors": len(errors),
        "model_requested": args.model,
        "models_returned": sorted({str(x.get("model_returned")) for x in api_ok}),
        "thinking_modes": dict(sorted(__import__("collections").Counter(x.get("thinking") for x in rows).items())),
        "temperature_values": dict(sorted(__import__("collections").Counter(str(x.get("temperature")) for x in rows).items())),
        "max_tokens_requested": dict(sorted(__import__("collections").Counter(str(x.get("max_tokens_requested")) for x in rows).items())),
        "exact_match": sum(bool(x["exact_match"]) for x in complete),
        "exact_match_rate_over_complete": sum(bool(x["exact_match"]) for x in complete) / len(complete) if complete else 0,
        "latency_seconds": {
            "mean": sum(x["latency_seconds"] for x in api_ok) / len(api_ok) if api_ok else None,
            "min": min((x["latency_seconds"] for x in api_ok), default=None),
            "max": max((x["latency_seconds"] for x in api_ok), default=None),
        },
        "usage": dict(usage),
        "note": "Smoke sample only; not a paper-level effectiveness result.",
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
