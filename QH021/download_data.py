from __future__ import annotations
import argparse
from pathlib import Path
parser=argparse.ArgumentParser(description="Download only the registered academic dataset source")
parser.add_argument("--root", type=Path, default=Path("./datasets"))
args=parser.parse_args()
args.root.mkdir(parents=True, exist_ok=True)
import urllib.request
url="https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
target=args.root/"raw/clevr/CLEVR_v1.0.zip"
target.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(url, target)
print(target)
