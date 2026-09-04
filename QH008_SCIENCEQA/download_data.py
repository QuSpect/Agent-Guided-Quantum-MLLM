from __future__ import annotations
import argparse
from pathlib import Path
parser=argparse.ArgumentParser(description="Download only the registered academic dataset source")
parser.add_argument("--root", type=Path, default=Path("./datasets"))
args=parser.parse_args()
args.root.mkdir(parents=True, exist_ok=True)
from huggingface_hub import snapshot_download
snapshot_download(repo_id="lmms-lab/ScienceQA", repo_type="dataset", revision="69dd4d6b67373d38f96a5badd5d24d0eb5bcdc50", local_dir=args.root/"raw/scienceqa")
