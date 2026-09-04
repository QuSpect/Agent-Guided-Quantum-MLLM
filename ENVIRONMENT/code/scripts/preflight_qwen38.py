#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoProcessor


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ID = "Qwen/Qwen3.8-27B"
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
MODEL_DIR = PROJECT_DIR / "models" / "Qwen3.8-27B"
RECORD_DIR = PROJECT_DIR / "records"


def main() -> None:
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    api = HfApi(endpoint=ENDPOINT)
    info = api.model_info(REPO_ID, files_metadata=True)
    siblings = []
    total_bytes = 0
    for item in info.siblings:
        size = int(item.size or 0)
        total_bytes += size
        siblings.append({"path": item.rfilename, "size": size})

    manifest = {
        "repo_id": REPO_ID,
        "endpoint": ENDPOINT,
        "revision": info.sha,
        "created_at": info.created_at.isoformat() if info.created_at else None,
        "last_modified": info.last_modified.isoformat() if info.last_modified else None,
        "total_bytes": total_bytes,
        "files": siblings,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    (RECORD_DIR / "qwen38_model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RECORD_DIR / "qwen38_model_revision.txt").write_text(info.sha + "\n", encoding="utf-8")

    metadata_patterns = [
        "*.json",
        "*.txt",
        "*.model",
        "*.tiktoken",
        "*.jinja",
        "*.py",
        "README.md",
        "LICENSE*",
        ".gitattributes",
    ]
    snapshot_download(
        repo_id=REPO_ID,
        revision=info.sha,
        endpoint=ENDPOINT,
        local_dir=MODEL_DIR,
        allow_patterns=metadata_patterns,
        max_workers=8,
    )

    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    result = {
        "status": "ok",
        "repo_id": REPO_ID,
        "revision": info.sha,
        "model_dir": str(MODEL_DIR),
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "processor_class": processor.__class__.__name__,
        "total_bytes": total_bytes,
        "endpoint": ENDPOINT,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    (RECORD_DIR / "qwen38_preflight.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

