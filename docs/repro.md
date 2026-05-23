# Reproduction Notes

## Goal
Reproduce a minimal RAMS-style few-shot ABSA pipeline with:
- local SR/SC synthesis substitutes,
- reliability modeling (RM),
- reliability-aware training (RAT).

## Default Experiment

```powershell
.\scripts\run_fewshot.ps1 -Dataset res -Shot 2 -Seed 42
```

The command writes:
- `outputs/res/2shot/<run_id>/train.log`
- `outputs/res/2shot/<run_id>/metrics.json`

## Suggested Matrix
- `lap`: 2-shot, 5-shot
- `res`: 2-shot, 5-shot
- `res15`: 2-shot, 5-shot
- `res16`: 2-shot, 5-shot

