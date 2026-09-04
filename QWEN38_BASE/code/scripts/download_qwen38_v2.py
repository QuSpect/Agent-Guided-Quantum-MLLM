#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ID = "Qwen/Qwen3.8-27B"
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
MODEL_DIR = PROJECT_DIR / "models" / "Qwen3.8-27B"
RECORD_DIR = PROJECT_DIR / "records"


def main() -> None:
    revision = (RECORD_DIR / "qwen38_model_revision.txt").read_text(encoding="utf-8").strip()
    started_at = datetime.now(timezone.utc).isoformat()
    path = snapshot_download(
        repo_id=REPO_ID,
        revision=revision,
        endpoint=ENDPOINT,
        local_dir=MODEL_DIR,
        max_workers=4,
    )
    result = {
        "status": "complete",
        "repo_id": REPO_ID,
        "revision": revision,
        "endpoint": ENDPOINT,
        "local_dir": path,
        "max_workers": 4,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (RECORD_DIR / "qwen38_download.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
