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

## Reporting Artifacts

- `project/results_summary.md`: concise claim-quality summary.
- `project/final_report_draft.md`: report scaffold with Thomas's current results and teammate placeholders.
- `project/teammate_result_template.csv`: copy/paste format for Brandy/Jacques result rows.
- `project/plots/defensible_throughput_summary.csv`: log-parsed stable-window metrics.
- `project/plots/*.svg`: generated figures for the write-up.

Regenerate figures and CSV summaries with:

```bash
python3 project/plot_benchmarks.py
```

## First Baselines

- Smoke test: `./launch.sh throughput 125m 2 1`
- Small baseline: `./launch.sh throughput 760m`
- Target baseline: `./launch.sh throughput 8b 50 1`
