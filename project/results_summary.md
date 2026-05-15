# Results Summary

This file records which benchmark numbers are suitable for project claims and which are only directional. Prefer the log-parsed metrics in `project/plots/defensible_throughput_summary.csv` over single final-iteration values from `project/benchmark_log.csv`.

## Claim Rules

- Use normal-partition runs for final throughput claims.
- Keep debug-partition runs separate; they are useful for smoke tests and hypotheses only.
- Prefer `avg_last_half` for runs with at least 20 iterations.
- Prefer `avg_excluding_first2` for very short runs.
- Treat failed, cancelled, timeout, and setup-only runs as non-claim rows even if they reached training.
- When a run is bursty, report both average and median from the stable window.

## Current Defensible 760m Findings

| Question | Normal-partition evidence | Conclusion |
|---|---:|---|
| Keep distributed optimizer? | `USE_DISTRIBUTED_OPTIMIZER=1`: avg last half `32565`; `USE_DISTRIBUTED_OPTIMIZER=0`: avg last half `32555` | No confirmed gain from disabling it; keep default on. |
| Best MBS among 2/4/8? | MBS 2: `16112`; MBS 4: `32565`; MBS 8: `14659` | Keep MBS 4 for 760m / 1 node / GBS 256. |
| Best GBS among 128/256/512? | GBS 128: avg last half `22743`; GBS 256 longer baseline: avg last half `43742`; GBS 512: avg last half `59626` | GBS 512 is the strongest candidate so far, but it needs replication or a longer run before making it the final headline claim. |
| Longer baseline reference | 50-step MBS 4 baseline: avg last half `43742`, median last half `42732` | Use as the cleaner 760m normal baseline, with variance caveat. |
| Normal NSYS profile? | Job `2235710` timed out during report generation; `.nsys-rep` is 0 bytes | Do not use it for profile claims. |

## Profiling Claims

The only usable NSYS profile summary so far is the short debug run `2227705`. It is directional, not final evidence. Its key result was that NCCL collectives dominated summed GPU kernel time, while CUDNN SDPA/flash attention was not the primary bottleneck.

For a normal-partition profile, rerun with lower overhead and enough report-finalization time, for example:

```bash
PARTITION=normal TIME_OVERRIDE=01:30:00 NSYS_TRACE=cuda,nvtx,nccl ./profile_nsys.sh 760m 4 1
```

## Generated Artifacts

- `project/plots/defensible_throughput_summary.csv`: log-parsed metrics, claim eligibility, iteration values.
- `project/plots/throughput_summary.csv`: lightweight run inventory for quick filtering.
- `project/plots/760m_normal_stable_throughput.svg`: normal-partition chart using stable-window metrics.
- `project/plots/760m_normal_stable_range.svg`: normal-partition min/median/max range chart for variability.
- `project/plots/760m_debug_stable_throughput.svg`: debug-partition chart, directional only.

Regenerate artifacts with:

```bash
python3 project/plot_benchmarks.py
```

## Report-Ready Thomas Contribution

Thomas's current contribution is a defensible throughput evaluation pipeline plus a first bottleneck diagnosis:

1. Updated the launcher so user-specific paths/accounts live in `config.sh`, not in generated scripts.
2. Added `USE_DISTRIBUTED_OPTIMIZER`, `MBS_OVERRIDE`, and `GBS_OVERRIDE` switches for controlled comparisons.
3. Added `profile_nsys.sh` and NSYS report/stat extraction for profiling runs.
4. Built log parsing that reports stable-window metrics and filters non-claim rows.
5. Ran controlled normal-partition comparisons for distributed optimizer and micro-batch size.

The strongest current claim is narrow but defensible: for the provided 760m, one-node, GBS 256 setup, MBS 4 with the default distributed optimizer remains the best measured configuration. Disabling the distributed optimizer did not reproduce the promising debug result on normal partition, and MBS 2/8 were clearly worse than MBS 4. The first normal GBS sweep suggests GBS 512 may be better than GBS 256, but that result should be replicated because normal-partition throughput is bursty.

## Teammate Integration Contract

When Brandy or Jacques add results, they should append rows to `project/benchmark_log.csv` using the same conventions:

| Field | Required convention |
|---|---|
| `feature` | Short stable slug, e.g. `cudnn-sdpa-normal-20s` or `fp8-normal-20s` |
| `status` | `completed`, `failed`, `timeout`, `cancelled`, or `submitted` |
| `notes` | Include command, relevant env vars, and whether the row is claim-quality |
| `tokens_per_sec_per_gpu` | Final/stable representative value, but plots will parse the raw log when available |
| `log_path` | Relative path under repo, e.g. `logs/<job>.log` |

After adding rows, run:

```bash
python3 project/plot_benchmarks.py
```

Then update this file with a short conclusion and whether the result is suitable for final claims.
