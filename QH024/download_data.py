from __future__ import annotations
import argparse
from pathlib import Path
parser=argparse.ArgumentParser(description="Download only the registered academic dataset source")
parser.add_argument("--root", type=Path, default=Path("./datasets"))
args=parser.parse_args()
args.root.mkdir(parents=True, exist_ok=True)
from datasets import load_dataset
revision="b08601e04326c79dfdd32d625aee71d232d685c3"
for split in ("train", "validation"):
    load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split, revision=revision, cache_dir=str(args.root/"cache"))
