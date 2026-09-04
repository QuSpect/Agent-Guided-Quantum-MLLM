from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
manifest_path=ROOT/"scheme_manifest.json"
if not manifest_path.exists():
    manifest_path=ROOT/"manifest.json"
manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
for row in manifest["parameters"]:
    src=ROOT/row["packaged_path"]
    dst=ROOT/row["runtime_path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size==row["size"]:
        continue
    shutil.copy2(src,dst)
print("runtime checkpoint layout ready under ./artifacts")
