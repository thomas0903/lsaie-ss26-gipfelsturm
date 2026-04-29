# Gipfelsturm Group Workflow

Challenge: maximum throughput, measured as tokens/sec/GPU.

## Branches

- Thomas: `thomas/profiling-compile`
- Brandy: `brandy/attention-backends`
- Jacques: `jacques/fp8-memory-fusion`

## Rules

- Record every run in `project/benchmark_log.csv`, including failed runs.
- Keep throughput tests short unless the group agrees otherwise.
- Use patches for Megatron-LM changes where possible.
- Do not queue long jobs without telling the group first.
- Compare every optimization against the same baseline command.

## First Baselines

- Smoke test: `./launch.sh throughput 125m 2 1`
- Small baseline: `./launch.sh throughput 760m`
- Target baseline: `./launch.sh throughput 8b 50 1`
