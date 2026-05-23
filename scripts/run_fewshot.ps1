param(
  [ValidateSet("lap","res","res15","res16")]
  [string]$Dataset = "res",
  [ValidateSet(2,5)]
  [int]$Shot = 2,
  [int]$Seed = 42,
  [string]$Config = "configs/base.yaml"
)

uv run rams-train --dataset $Dataset --shot $Shot --seed $Seed --config $Config

