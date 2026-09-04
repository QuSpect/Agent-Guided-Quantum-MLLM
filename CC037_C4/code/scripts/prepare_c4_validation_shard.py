#!/usr/bin/env python3
"""Download one pinned C4 validation shard and tokenize a deterministic prefix."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer


PROJECT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT / "datasets/raw/c4_validation"
PROCESSED_DIR = PROJECT / "datasets/processed/c4_validation_qwen38"
OUTPUT = PROJECT / "artifacts/c4-validation-preparation.json"
REPO = "allenai/c4"
FILENAME = "en/c4-validation.00000-of-00008.json.gz"
ROW_LIMIT = 2048


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    endpoint = "https://hf-mirror.com"
    revision = HfApi(endpoint=endpoint).dataset_info(REPO).sha
    downloaded = Path(hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=FILENAME,
        revision=revision,
        endpoint=endpoint,
    ))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    subset_path = RAW_DIR / "c4-validation-shard0-prefix2048.jsonl"
    texts: list[str] = []
    with gzip.open(downloaded, "rt", encoding="utf-8") as source, subset_path.open(
        "w", encoding="utf-8"
    ) as target:
        for row_index, line in enumerate(source):
            if row_index >= ROW_LIMIT:
                break
            row = json.loads(line)
            text = str(row["text"])
            texts.append(text)
            target.write(json.dumps({
                "source_row_index": row_index,
                "text": text,
                "url": row.get("url"),
                "timestamp": row.get("timestamp"),
            }, ensure_ascii=False) + "\n")
    if len(texts) != ROW_LIMIT:
        raise RuntimeError(f"expected {ROW_LIMIT} rows, got {len(texts)}")
    model_dir = Path((PROJECT / "records/active_model_path.txt").read_text().strip())
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.model_max_length = 10**12
    token_ids = torch.tensor(
        tokenizer("\n\n".join(texts), add_special_tokens=False)["input_ids"], dtype=torch.int32
    )
    token_path = PROCESSED_DIR / "validation-tokens.pt"
    torch.save(token_ids, token_path)
    payload = {
        "status": "ok",
        "dataset": REPO,
        "config": "en",
        "source_revision": revision,
        "license": "ODC-BY 1.0; underlying Common Crawl terms also apply",
        "split": "validation",
        "source_file": FILENAME,
        "source_file_sha256": sha256(downloaded),
        "deterministic_subset_rule": f"first {ROW_LIMIT} rows of pinned validation shard 0",
        "rows": len(texts),
        "subset_path": str(subset_path),
        "subset_sha256": sha256(subset_path),
        "token_path": str(token_path),
        "token_sha256": sha256(token_path),
        "token_count": int(token_ids.numel()),
        "tokenizer_path": str(model_dir),
        "test_split_exists_or_was_requested_materialized_inspected_or_used": 0,
        "train_split_requested_materialized_inspected_or_used": 0,
        "purpose": "untouched external validation for true-layer replacement search",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
