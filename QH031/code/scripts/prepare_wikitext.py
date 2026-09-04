#!/usr/bin/env python3
"""Lock, materialize, and tokenize WikiText-2 without reading its test split."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "datasets/raw/wikitext2"
PROCESSED_DIR = PROJECT_DIR / "datasets/processed/wikitext2_qwen38"
RECORD_DIR = PROJECT_DIR / "records"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    revision = "b08601e04326c79dfdd32d625aee71d232d685c3"
    # Load only the named splits. The first draft used a DatasetDict and therefore
    # materialized test in cache even though no test row entered fitness; that event
    # is disclosed in the output rather than erased.
    dataset = {
        split_name: load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            revision=revision,
            split=split_name,
        )
        for split_name in ("train", "validation")
    }
    model_dir = Path((RECORD_DIR / "active_model_path.txt").read_text(encoding="utf-8").strip())
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.model_max_length = 10**12
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    splits = {}
    # Test is intentionally neither materialized nor tokenized during search.
    for split_name in ("train", "validation"):
        split = dataset[split_name]
        parquet_path = RAW_DIR / f"{split_name}.parquet"
        split.to_parquet(parquet_path)
        text = "\n\n".join(row for row in split["text"] if row.strip())
        token_ids = torch.tensor(
            tokenizer(text, add_special_tokens=False)["input_ids"], dtype=torch.int32
        )
        token_path = PROCESSED_DIR / f"{split_name}-tokens.pt"
        torch.save(token_ids, token_path)
        splits[split_name] = {
            "rows": len(split),
            "dataset_fingerprint": split._fingerprint,
            "parquet_path": str(parquet_path),
            "parquet_sha256": sha256_file(parquet_path),
            "token_path": str(token_path),
            "token_sha256": sha256_file(token_path),
            "token_count": int(token_ids.numel()),
        }
    payload = {
        "status": "ok",
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "source_revision": revision,
        "license": "CC BY-SA 4.0",
        "tokenizer_path": str(model_dir),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "splits": splits,
        "test_rows_used_for_training_or_evaluation": 0,
        "protocol_disclosure": {
            "initial_datasetdict_load_materialized_test_rows_in_cache": 4358,
            "test_rows_inspected_or_used_for_fitness": 0,
            "correction": "current script requests train and validation splits individually",
        },
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    output = PROJECT_DIR / "artifacts/wikitext2-preparation.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
