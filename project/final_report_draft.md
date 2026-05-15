# Gipfelsturm Final Report Draft

## Project Direction

We target Challenge 2: maximizing training throughput, measured as tokens/sec/GPU, for Megatron-LM training on Clariden. Our proposal focused on the single-GPU tier around the 8B model with a stretch toward single-node scaling. The working evaluation in this repository currently centers on the 760m, one-node setup because it is fast enough for controlled experiments while still exercising the same Megatron-LM, TransformerEngine, data loading, and distributed optimizer path.

## Contributions

### Thomas: Profiling, Benchmark Harness, And Synthesis

Thomas built the measurement backbone:

- launcher controls for `USE_DISTRIBUTED_OPTIMIZER`, `MBS_OVERRIDE`, `GBS_OVERRIDE`, `PARTITION`, and `TIME_OVERRIDE`;
- an NSYS profiling wrapper through `profile_nsys.sh`;
- log parsing for stable-window throughput metrics;
- normal/debug result separation;
- benchmark plots under `project/plots/`;
- this summary/report scaffold.

Current Thomas-side conclusion:

- The 760m, one-node, GBS 256 baseline is noisy on normal partition, so single final-iteration values are not reliable enough for claims.
- Stable-window metrics show that MBS 4 is the best tested micro-batch size among 2, 4, and 8.
- The first normal-partition GBS sweep suggests GBS 512 is a promising next baseline, but it should be replicated or run longer before using it as the strongest claim.
- Disabling the distributed optimizer did not improve the normal-partition result.
- The usable debug NSYS profile suggests NCCL collectives dominate summed GPU kernel time; attention is already using CUDNN/flash SDPA and is not the obvious first bottleneck for Thomas's lane.

### Brandy: Attention Backend Experiments

Status: waiting for results.

Expected table:

| Feature | Command | Job ID | Model | Nodes | MBS | GBS | Stable tokens/sec/GPU | Status | Notes |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

What to add once available:

- backend used before/after;
- exact patch or config change;
- whether the run is normal-partition claim-quality;
- any correctness/stability issues.

### Jacques: FP8, Memory, Fused Ops, Batch Sweeps

Status: waiting for results.

Expected table:

| Feature | Command | Job ID | Model | Nodes | MBS | GBS | Stable tokens/sec/GPU | Status | Notes |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

What to add once available:

- precision mode and TransformerEngine settings;
- memory footprint changes;
- maximum stable batch size;
- whether fused kernels or offloading changed throughput.

## Methodology

All runs are logged in `project/benchmark_log.csv`. We distinguish between:

- normal-partition runs, used for final throughput claims;
- debug-partition runs, used only for smoke tests and directional hypotheses;
- failed, cancelled, or timed-out runs, retained to document negative results and avoid repeating bad experiments.

The final report should use `project/plots/defensible_throughput_summary.csv`, not raw final-iteration values. The plotting script parses raw logs and computes:

- `avg_excluding_first2` for short runs;
- `avg_last_half` for runs with at least 20 iterations;
- medians and min/max ranges to expose burstiness.

This is necessary because normal-partition throughput is visibly bursty. A single final iteration can overstate or understate the stable behavior.

## Current Results

### Normal-Partition 760m Results

Use `project/plots/760m_normal_stable_throughput.svg` for the compact bar chart and `project/plots/760m_normal_stable_range.svg` for variability.

Key Thomas-side rows:

| Experiment | Stable metric | Interpretation |
|---|---:|---|
| MBS 2, GBS 256 | 16112 | Too small per-GPU work; worse throughput |
| MBS 4, GBS 256 | 32565 | Best controlled 20-step baseline |
| No distributed optimizer, MBS 4, GBS 256 | 32555 | Essentially tied with baseline; no confirmed win |
| MBS 8, GBS 256 | 14659 | Worse than baseline |
| Longer MBS 4 baseline | 43742 | Cleaner 50-step reference, but bursty |
| MBS 4, GBS 128 | 22743 | Lower GBS is worse in the 30-step sweep |
| MBS 4, GBS 512 | 59626 | Best observed stable-window result so far; replicate or run longer before final headline claim |

### Profiling

The usable profile is a short debug run:

```bash
PARTITION=debug TIME_OVERRIDE=00:15:00 ./profile_nsys.sh 760m 4 1
```

It produced a valid report and CSV stats. The grouped GPU kernel time was:

| Group | Share |
|---|---:|
| NCCL collectives | 53.9% |
| TransformerEngine / GEMM kernels | 23.8% |
| CUDNN SDPA / flash attention | 12.1% |
| Other Triton kernels | 5.1% |
| Other / norm kernels | 5.0% |

This is directional because it ran on debug and includes warmup. The normal-partition NSYS attempt completed training but timed out while generating the report, leaving a 0-byte `.nsys-rep`; it should not be used for profiling claims.

## Figures To Include

- `project/plots/760m_normal_stable_throughput.svg`
- `project/plots/760m_normal_stable_range.svg`
- `project/plots/760m_debug_stable_throughput.svg` only if the text clearly labels it as directional

## Open Work

- Integrate Brandy's attention backend results.
- Integrate Jacques's FP8/memory/fusion results.
- Replicate the GBS 512 result or run a longer 50-step GBS 512 baseline if time permits.
- If a final profile is required, rerun NSYS with lower overhead and enough report-generation time:

```bash
PARTITION=normal TIME_OVERRIDE=01:30:00 NSYS_TRACE=cuda,nvtx,nccl ./profile_nsys.sh 760m 4 1
```
