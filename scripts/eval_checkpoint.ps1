param(
  [ValidateSet("lap","res","res15","res16")]
  [string]$Dataset = "res",
  [ValidateSet("dev","test")]
  [string]$Split = "test",
  [Parameter(Mandatory = $true)]
  [string]$CheckpointDir,
  [string]$Config = "configs/base.yaml"
)

uv run rams-eval --dataset $Dataset --split $Split --ckpt $CheckpointDir --config $Config

