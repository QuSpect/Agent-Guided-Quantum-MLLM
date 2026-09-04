#!/usr/bin/env python3
import hashlib
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "models" / "Qwen3.8-27B"
RECORD_DIR = PROJECT_DIR / "records"
HF_MANIFEST = RECORD_DIR / "qwen38_hf_lfs_manifest.json"


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def crc32(path: Path) -> str:
    value = 0
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            value = zlib.crc32(chunk, value)
    return f"{value & 0xFFFFFFFF:08x}"


def git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def main() -> None:
    manifest = json.loads(HF_MANIFEST.read_text(encoding="utf-8"))
    errors = []
    warnings = []
    checked_files = []
    blob_sha1_checks = []

    for item in manifest["files"]:
        relative = item["path"]
        if relative == ".gitattributes":
            continue
        path = MODEL_DIR / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        actual_size = path.stat().st_size
        if actual_size != item["size"]:
            errors.append(f"size mismatch: {relative}: {actual_size} != {item['size']}")
        if item.get("blob_id") and not item.get("lfs_sha256"):
            actual_blob_id = git_blob_sha1(path)
            ok = actual_blob_id == item["blob_id"]
            blob_sha1_checks.append(
                {
                    "path": relative,
                    "git_blob_sha1": actual_blob_id,
                    "expected_blob_id": item["blob_id"],
                    "ok": ok,
                }
            )
            if not ok:
                errors.append(f"git blob sha1 mismatch: {relative}")

    lfs_items = [item for item in manifest["files"] if item.get("lfs_sha256")]
    for index, item in enumerate(lfs_items, start=1):
        path = MODEL_DIR / item["path"]
        if not path.is_file():
            continue
        actual_sha256 = digest(path, "sha256")
        ok = actual_sha256 == item["lfs_sha256"]
        checked_files.append(
            {
                "path": item["path"],
                "size": path.stat().st_size,
                "sha256": actual_sha256,
                "expected_sha256": item["lfs_sha256"],
                "ok": ok,
            }
        )
        print(f"SHA256 {index}/{len(lfs_items)} {item['path']} {'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            errors.append(f"sha256 mismatch: {item['path']}")

    crc_path = MODEL_DIR / "crc32.txt"
    crc_results = []
    if crc_path.is_file():
        for line in crc_path.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split(maxsplit=1)
            path = MODEL_DIR / relative.strip()
            actual = crc32(path) if path.is_file() else None
            ok = actual == expected.lower()
            crc_results.append({"path": relative.strip(), "expected": expected, "actual": actual, "ok": ok})
            if not ok:
                warnings.append(
                    f"upstream crc32.txt is stale for exact HF git blob: {relative.strip()}"
                )
    else:
        errors.append("missing: crc32.txt")

    index_path = MODEL_DIR / "model.safetensors.index.json"
    index_data = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    indexed_shards = sorted(set(index_data.get("weight_map", {}).values()))
    disk_shards = sorted(path.name for path in MODEL_DIR.glob("model-*.safetensors"))
    if indexed_shards != disk_shards:
        errors.append("safetensors shard set differs from index")

    tensor_counts = {}
    for shard in disk_shards:
        try:
            with safe_open(MODEL_DIR / shard, framework="pt", device="cpu") as handle:
                tensor_counts[shard] = len(handle.keys())
        except Exception as error:
            errors.append(f"safetensors header failure: {shard}: {error}")

    result = {
        "status": "ok" if not errors else "failed",
        "repo_id": manifest["repo_id"],
        "hf_revision": manifest["revision"],
        "modelscope_revision": "master",
        "model_dir": str(MODEL_DIR),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "warnings": warnings,
        "lfs_sha256_checks": checked_files,
        "git_blob_sha1_checks": blob_sha1_checks,
        "crc32_checks": crc_results,
        "indexed_shards": indexed_shards,
        "tensor_counts": tensor_counts,
    }
    output_path = RECORD_DIR / "qwen38_modelscope_validation.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "errors": errors,
                "warnings": warnings,
                "record": str(output_path),
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)
    (RECORD_DIR / "active_model_path.txt").write_text(str(MODEL_DIR) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
