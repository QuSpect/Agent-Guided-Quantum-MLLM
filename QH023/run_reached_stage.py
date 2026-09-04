from __future__ import annotations
import json, os, runpy, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PLAN=json.loads('{"preprocess": ["code/scripts/prepare_editclevr_dev.py"], "stage": [["code/scripts/run_editclevr_qh023.py", "--candidate", "base"], ["code/scripts/run_editclevr_qh023.py", "--candidate", "qh023"], ["code/scripts/run_editclevr_qh023.py", "--candidate", "dq023"], ["code/scripts/run_editclevr_qh023.py", "--candidate", "cc023"], ["code/scripts/run_editclevr_qh023.py", "--candidate", "qh023_noent"]], "analysis": ["code/scripts/analyze_editclevr_qh023.py"], "entrypoint": "run_editclevr_qh023.py"}')
base=ROOT/"models/Qwen3.8-27B"
if not base.exists():
    raise FileNotFoundError("Run: python download_base_model.py --output ./models/Qwen3.8-27B")
os.chdir(ROOT)
runpy.run_path(str(ROOT/"prepare_scheme_weights.py"), run_name="__main__")
sys.path.insert(0,str(ROOT/"code/src"))
os.environ.setdefault("HF_HOME",str(ROOT/".cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE",str(ROOT/".cache/huggingface"))
def execute(parts):
    command=[sys.executable,str(ROOT/parts[0]),*parts[1:]]
    print("$ "+" ".join(command),flush=True)
    subprocess.run(command,cwd=ROOT,check=True)
for script in PLAN["preprocess"]:
    execute([script])
for command in PLAN["stage"]:
    execute(command)
for script in PLAN["analysis"]:
    execute([script])
if not PLAN["stage"]:
    print("No standalone stage runner exists for this support-only archive unit.")
