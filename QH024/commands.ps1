$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
python -m pip install -r requirements.txt
python download_base_model.py --output ./models/Qwen3.8-27B
python download_data.py --root ./datasets
python prepare_scheme_weights.py
python run_reached_stage.py
