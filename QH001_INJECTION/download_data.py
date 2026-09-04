from __future__ import annotations
import argparse
from pathlib import Path
parser=argparse.ArgumentParser(description="Download only the registered academic dataset source")
parser.add_argument("--root", type=Path, default=Path("./datasets"))
args=parser.parse_args()
args.root.mkdir(parents=True, exist_ok=True)
print("This experiment unit has no task dataset download stage.")
