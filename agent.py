"""OpenAI-backed grouping harness for scan JSONL output."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_MERGE_GAP = 2.0


@dataclass
class CanonicalResult:
    input_id: str
    anchor_present: bool
    canonical_label: str
    normalized_line: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_json(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    cleaned = "".join(ch.casefold() if ch.isalnum() else " " for ch in value)
    return " ".join(cleaned.split())


def load_dotenv(path: Path, *, override: bool = True) -> int:
    if not path.exists():
        return 0

    loaded = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or (key in os.environ and not override):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value
        loaded += 1
    return loaded


def _line_key(row: dict) -> str:
    return _normalize_text(str(row.get("text", "")))


def _fallback_match_id(row: dict) -> str:
    return _hash_json({
        "path": row.get("path", ""),
        "timestamp": row.get("timestamp", ""),
        "text": row.get("text", ""),
        "confidence": row.get("confidence", ""),
    })


def _output_paths(input_path: Path, output: Optional[str]) -> tuple[Path, Path, Path]:
    if output:
        stem = Path(output)
    elif input_path.is_dir():
        stem = Path(input_path.name)
    elif input_path.suffix.lower() == ".jsonl":
        stem = input_path.with_suffix("")
    else:
        stem = input_path
    return (
        Path(f"{stem}.agent.jsonl"),
        Path(f"{stem}.agent.json"),
        Path(f"{stem}.agent.state.json"),
    )


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping invalid JSON in {path}:{line_number}: {exc}", file=sys.stderr)
                continue
            record.setdefault("_input_jsonl", str(path))
            rows.append(record)
    return rows


def _iter_jsonl_paths(input_path: Path) -> Iterable[Path]:
    if input_path.is_dir():
        yield from sorted(
            p for p in input_path.rglob("*.jsonl")
            if not p.name.endswith(".agent.jsonl")
        )
    elif input_path.suffix.lower() == ".csv":
        with input_path.open("r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw = (row.get("jsonl_path") or "").strip()
                if raw:
                    yield Path(raw)
    else:
        yield input_path


def _read_input_rows(input_path: Path) -> list[dict]:
    rows: list[dict] = []
    for jsonl_path in _iter_jsonl_paths(input_path):
        if not jsonl_path.exists():
            print(f"Warning: JSONL not found: {jsonl_path}", file=sys.stderr)
            continue
        rows.extend(_read_jsonl(jsonl_path))
    return rows


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"canonical_cache": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"canonical_cache": {}}
    if not isinstance(data, dict):
        return {"canonical_cache": {}}
    data.setdefault("canonical_cache", {})
    return data


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _chunks(items: list[dict], size: int) -> Iterable[list[dict]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def _canonical_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "input_id": {"type": "string"},
                        "anchor_present": {"type": "boolean"},
                        "canonical_label": {"type": "string"},
                        "normalized_line": {"type": "string"},
                    },
                    "required": [
                        "input_id",
                        "anchor_present",
                        "canonical_label",
                        "normalized_line",
                    ],
                },
            }
        },
        "required": ["items"],
    }


def _build_openai_payload(model: str, search_term: str, items: list[dict]) -> dict:
    system_prompt = (
        "You canonicalize OCR text lines from video frames. The search term is "
        "an anchor that identifies relevant lines. For each OCR line, decide "
        "whether the anchor is present despite capitalization, spacing, or minor "
        "OCR noise. If present, return a concise canonical label for the text "
        "beside the anchor. Do not decide timestamps or grouping boundaries."
    )
    return {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": json.dumps({
                        "search_term": search_term,
                        "items": items,
                    }, ensure_ascii=False),
                }],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ocr_line_canonicalization",
                "strict": True,
                "schema": _canonical_schema(),
            }
        },
    }


def _extract_response_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    for output in data.get("output", []):
        for content in output.get("content", []):
            if isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("OpenAI response did not contain text output")


class OpenAICanonicalizer:
    def __init__(self, *, api_key: str, model: str, timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def canonicalize(self, *, search_term: str, items: list[dict]) -> list[CanonicalResult]:
        payload = _build_openai_payload(self.model, search_term, items)
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI request failed ({exc.code}): {detail[:1000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        data = json.loads(body)
        parsed = json.loads(_extract_response_text(data))
        results = []
        for item in parsed.get("items", []):
            results.append(CanonicalResult(
                input_id=str(item["input_id"]),
                anchor_present=bool(item["anchor_present"]),
                canonical_label=str(item["canonical_label"]).strip(),
                normalized_line=str(item["normalized_line"]).strip(),
            ))
        return results


def _canonicalize_missing(
    rows: list[dict],
    *,
    search_term: str,
    canonicalizer: OpenAICanonicalizer,
    cache: dict,
    batch_size: int,
) -> int:
    pending_by_key: dict[str, dict] = {}
    for row in rows:
        key = _line_key(row)
        if key and key not in cache:
            pending_by_key[key] = {
                "input_id": key,
                "text": str(row.get("text", "")),
                "normalized_text": key,
            }

    completed = 0
    for batch in _chunks(list(pending_by_key.values()), batch_size):
        for result in canonicalizer.canonicalize(search_term=search_term, items=batch):
            cache[result.input_id] = {
                "anchor_present": result.anchor_present,
                "canonical_label": result.canonical_label,
                "normalized_line": result.normalized_line,
            }
            completed += 1
    return completed


def _match_id(row: dict) -> str:
    return str(row.get("match_id") or _fallback_match_id(row))


def _row_sort_key(row: dict) -> tuple:
    return (
        str(row.get("path", "")),
        str(row.get("scan_id", "")),
        float(row.get("timestamp", 0.0)),
        _match_id(row),
    )


def _append_group(groups: list[dict], current: list[dict], label: str) -> None:
    if not current:
        return
    start = float(current[0]["timestamp"])
    end = float(current[-1]["timestamp"])
    confidence_sum = sum(float(row.get("confidence", 0.0)) for row in current)
    match_ids = [_match_id(row) for row in current]
    sample_texts = list(dict.fromkeys(str(row.get("text", "")) for row in current))
    first = current[0]
    scan_ids = list(dict.fromkeys(str(row.get("scan_id", "")) for row in current if row.get("scan_id")))
    groups.append({
        "group_id": _hash_json({
            "source_id": first.get("source_id", first.get("path", "")),
            "scan_id": first.get("scan_id", ""),
            "canonical_text": label,
            "start_timestamp": f"{start:.3f}",
            "end_timestamp": f"{end:.3f}",
            "first_match_id": match_ids[0],
            "last_match_id": match_ids[-1],
        }),
        "source_id": first.get("source_id", ""),
        "scan_id": first.get("scan_id", ""),
        "scan_ids": scan_ids,
        "path": first.get("path", ""),
        "file": first.get("file", ""),
        "canonical_text": label,
        "sample_texts": sample_texts,
        "start_timestamp": start,
        "end_timestamp": end,
        "match_count": len(current),
        "average_confidence": round(confidence_sum / len(current), 4),
        "first_match_id": match_ids[0],
        "last_match_id": match_ids[-1],
        "match_ids": match_ids,
    })


def group_rows(rows: list[dict], canonical_cache: dict, *, merge_gap: float) -> list[dict]:
    deduped: dict[str, dict] = {}
    for row in rows:
        deduped.setdefault(_match_id(row), row)

    groups: list[dict] = []
    current: list[dict] = []
    current_label = ""

    for row in sorted(deduped.values(), key=_row_sort_key):
        result = canonical_cache.get(_line_key(row), {})
        if not result.get("anchor_present"):
            continue
        label = str(result.get("canonical_label", "")).strip()
        if not label:
            continue

        if not current:
            current = [row]
            current_label = label
            continue

        previous = current[-1]
        same_group = (
            str(previous.get("source_id", previous.get("path", ""))) == str(row.get("source_id", row.get("path", "")))
            and str(previous.get("scan_id", "")) == str(row.get("scan_id", ""))
            and current_label == label
            and float(row["timestamp"]) - float(previous["timestamp"]) <= merge_gap
        )
        if same_group:
            current.append(row)
        else:
            _append_group(groups, current, current_label)
            current = [row]
            current_label = label

    _append_group(groups, current, current_label)
    return groups


def _write_outputs(
    *,
    jsonl_path: Path,
    meta_path: Path,
    groups: list[dict],
    started_at: str,
    input_path: Path,
    row_count: int,
    options: dict,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for group in groups:
            fh.write(json.dumps(group, ensure_ascii=False) + "\n")

    metadata = {
        "input": str(input_path),
        "started_at": started_at,
        "completed_at": _now(),
        "match_count": row_count,
        "group_count": len(groups),
        "options": options,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _process_once(args, *, started_at: str) -> tuple[int, int]:
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: not found: {input_path}")

    load_dotenv(Path(args.env_file), override=not args.no_env_override)

    out_jsonl, out_meta, state_path = _output_paths(input_path, args.output)
    state = _load_state(state_path)
    cache = state["canonical_cache"]
    rows = _read_input_rows(input_path)
    if args.force:
        cache.clear()

    missing = {_line_key(row) for row in rows if _line_key(row) and _line_key(row) not in cache}
    if missing:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit(
                "Error: OPENAI_API_KEY is required for uncached OCR text canonicalization."
            )
        canonicalizer = OpenAICanonicalizer(api_key=api_key, model=args.openai_model)
        try:
            completed = _canonicalize_missing(
                rows,
                search_term=args.search_term,
                canonicalizer=canonicalizer,
                cache=cache,
                batch_size=args.batch_size,
            )
        except RuntimeError as exc:
            sys.exit(f"Error: {exc}")
        state["updated_at"] = _now()
        _write_state(state_path, state)
        print(f"Canonicalized {completed} OCR line variant(s).", flush=True)

    groups = group_rows(rows, cache, merge_gap=args.merge_gap)
    _write_outputs(
        jsonl_path=out_jsonl,
        meta_path=out_meta,
        groups=groups,
        started_at=started_at,
        input_path=input_path,
        row_count=len(rows),
        options={
            "search_term": args.search_term,
            "merge_gap": args.merge_gap,
            "openai_model": args.openai_model,
            "batch_size": args.batch_size,
        },
    )
    return len(rows), len(groups)


def run_agent(args) -> None:
    started_at = _now()
    while True:
        match_count, group_count = _process_once(args, started_at=started_at)
        print(f"Wrote {group_count} group(s) from {match_count} match row(s).", flush=True)
        if not args.watch:
            return
        time.sleep(args.poll_interval)
