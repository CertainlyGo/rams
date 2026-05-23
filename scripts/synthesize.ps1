param(
  [ValidateSet("lap","res","res15","res16")]
  [string]$Dataset = "res",
  [ValidateSet(2,5)]
  [int]$Shot = 2,
  [int]$Seed = 42,
  [int]$Multiplier = 1,
  [string]$Output = "outputs/synth.json"
)

uv run rams-synthesize --dataset $Dataset --shot $Shot --seed $Seed --multiplier $Multiplier --output $Output

