#!/usr/bin/env python3
"""Download only EditCLEVR development archives at the prelocked revision."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT = Path(__file__).resolve().parents[2]
LOCK = PROJECT / "artifacts/editclevr-hf-metadata-lock.json"
DESTINATION = PROJECT / "datasets/raw/editclevr-dev"
ALLOWED = ["README.md", "editclevr_atomic_id.tar.gz", "editclevr_splits.tar.gz"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    expected = {item["path"]: item for item in lock["siblings"]}
    snapshot_download(
        repo_id=lock["repo_id"],
        repo_type="dataset",
        revision=lock["revision"],
        allow_patterns=ALLOWED,
        local_dir=DESTINATION,
        max_workers=8,
    )
    files = []
    for relative in ALLOWED:
        path = DESTINATION / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        expected_digest = expected[relative]["lfs_sha256"]
        if expected_digest is not None and digest != expected_digest:
            raise RuntimeError(f"SHA-256 mismatch for {relative}")
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    forbidden = [
        "editclevr_cogent_ood.tar.gz",
        "editclevr_hard_distractor.tar.gz",
        "editclevr_no_edit.tar.gz",
    ]
    if any((DESTINATION / name).exists() for name in forbidden):
        raise RuntimeError("blind test archive was unexpectedly downloaded")
    receipt = {
        "status": "development_archives_downloaded_and_verified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": lock["repo_id"],
        "revision": lock["revision"],
        "files": files,
        "forbidden_test_archives_absent": True,
        "test_images_or_annotations_read": 0,
    }
    output = PROJECT / "artifacts/editclevr-dev-download-receipt.json"
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
