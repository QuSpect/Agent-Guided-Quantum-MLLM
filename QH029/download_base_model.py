from __future__ import annotations
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

parser=argparse.ArgumentParser(description="Download the pinned official Qwen3.8-27B snapshot")
parser.add_argument("--output", type=Path, default=Path("./models/Qwen3.8-27B"))
parser.add_argument("--endpoint", default=None, help="Optional mirror, e.g. https://hf-mirror.com")
args=parser.parse_args()
if args.endpoint:
    import os
    os.environ["HF_ENDPOINT"]=args.endpoint
args.output.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id="Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", local_dir=args.output)
print(args.output.resolve())
