# RAMS-ABSA (Minimal Reproduction)

This repository provides a minimal, reproducible implementation of a RAMS-style few-shot ABSA pipeline over:
- `lap`
- `res`
- `res15`
- `res16`

## Quick Start

1. Create Python 3.12.4 environment and install deps:

```powershell
uv python install 3.12.4
uv venv --python 3.12.4
uv sync
```

2. Run few-shot train:

```powershell
.\scripts\run_fewshot.ps1 -Dataset res -Shot 2 -Seed 42
```

3. Evaluate checkpoint:

```powershell
.\scripts\eval_checkpoint.ps1 -Dataset res -Split test -CheckpointDir outputs\res\2shot\<run_id>\checkpoints
```

4. Generate synthetic samples:

```powershell
.\scripts\synthesize.ps1 -Dataset res -Shot 2 -Multiplier 1 -Output outputs\res_synth.json
```

## CLI

- `rams-train --dataset --shot --config --seed`
- `rams-eval --dataset --split --ckpt --config`
- `rams-synthesize --dataset --shot --seed --multiplier --output`

## Output Layout

`outputs/{dataset}/{shot}shot/{run_id}/`
- `metrics.json`
- `train.log`
- `checkpoints/`
