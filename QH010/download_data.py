from __future__ import annotations
import argparse
from pathlib import Path
parser=argparse.ArgumentParser(description="Download only the registered academic dataset source")
parser.add_argument("--root", type=Path, default=Path("./datasets"))
args=parser.parse_args()
args.root.mkdir(parents=True, exist_ok=True)
from huggingface_hub import snapshot_download
snapshot_download(repo_id="facebook/textvqa", repo_type="dataset", revision="30a47cfa557c996f64903f01250799d453be6215", local_dir=args.root/"raw/textvqa")
