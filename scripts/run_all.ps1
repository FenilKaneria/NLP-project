# Repeats the whole project from the raw data to the result tables.
# Run from the project root with the virtual environment active:
#     .\scripts\run_all.ps1
# Takes about 5 hours on an RTX 3050. Add -SkipSeeds to leave out the extra seeds.

param([switch]$SkipSeeds)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

function Run {
    Write-Host ">> python -m $args" -ForegroundColor Cyan
    python -m @args
    if ($LASTEXITCODE -ne 0) { throw "step failed: $args" }
}

# 1. Data
Run hinglish_mt.splits
Run hinglish_mt.analysis
Run hinglish_mt.synth
Run hinglish_mt.tokenise

# 2. Training
Run hinglish_mt.train configs/run_a.yaml
Run hinglish_mt.train configs/run_b.yaml
Run hinglish_mt.train configs/pretrain_clean.yaml
Run hinglish_mt.train configs/pretrain_noisy.yaml
Run hinglish_mt.train configs/run_c.yaml
Run hinglish_mt.train configs/run_d.yaml

# Run E: copy mechanism. Run F: copy mechanism and code-mixed synthetic data.
Run hinglish_mt.train configs/pretrain_noisy_copy.yaml
Run hinglish_mt.train configs/run_e.yaml
Run hinglish_mt.codemix
Run hinglish_mt.train configs/pretrain_mixed_copy.yaml
Run hinglish_mt.train configs/run_f.yaml

$runs = @("copy", "run_a", "run_b", "run_c", "run_d", "run_e", "run_f")

if (-not $SkipSeeds) {
    foreach ($seed in 43, 44) {
        foreach ($run in "run_b", "run_c", "run_d", "run_e", "run_f") {
            Run hinglish_mt.train "configs/$run.yaml" --set "name=${run}_seed$seed" "seed=$seed"
            $runs += "${run}_seed$seed"
        }
    }
}

# 3. Scores and tables
Run hinglish_mt.evaluate @runs
Run hinglish_mt.ablation
Run hinglish_mt.buckets
Run hinglish_mt.rare_words run_b run_d run_e run_f
Run hinglish_mt.embeddings
if (Test-Path "data/SentiMix/train_14k_split_conll.txt") {
    Run hinglish_mt.sentiment
}
Run hinglish_mt.error_sheets
