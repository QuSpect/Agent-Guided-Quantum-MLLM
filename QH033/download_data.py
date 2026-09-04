from __future__ import annotations
import argparse
from pathlib import Path
parser=argparse.ArgumentParser(description="Download only the registered academic dataset source")
parser.add_argument("--root", type=Path, default=Path("./datasets"))
args=parser.parse_args()
args.root.mkdir(parents=True, exist_ok=True)
from huggingface_hub import hf_hub_download
path=hf_hub_download(repo_id="allenai/c4", repo_type="dataset", revision="1588ec454efa1a09f29cd18ddd04fe05fc8653a2", filename="en/c4-validation.00000-of-00008.json.gz", local_dir=args.root/"raw/c4")
print(path)
