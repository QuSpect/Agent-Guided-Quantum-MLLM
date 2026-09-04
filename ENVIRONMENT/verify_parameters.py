from __future__ import annotations
import hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest_path=root/"scheme_manifest.json"
if not manifest_path.exists():
    manifest_path=root/"manifest.json"
manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
issues=[]
for row in manifest.get("parameters",[]):
    path=root/row["packaged_path"]
    if not path.exists():
        issues.append({"path":row["packaged_path"],"error":"missing"})
        continue
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    if path.stat().st_size!=row["size"] or h.hexdigest()!=row["sha256"]:
        issues.append({"path":row["packaged_path"],"error":"size_or_sha256_mismatch"})
print(json.dumps({"files":len(manifest.get("parameters",[])),"issues":issues,"pass":not issues},ensure_ascii=False,indent=2))
raise SystemExit(1 if issues else 0)
