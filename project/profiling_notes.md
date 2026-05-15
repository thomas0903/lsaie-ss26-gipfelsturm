# Profiling Notes

## 2026-05-14: First NSYS Profile

Command:

```bash
PARTITION=debug TIME_OVERRIDE=00:15:00 ./profile_nsys.sh 760m 4 1
```

Run:
- Job: `2227705`
- Commit: `d4f20ff`
- Branch: `thomas/profiling-compile`
- Model: `760m`
- Nodes / GPUs: `1 / 4`
- Steps: `4`
- Log: `logs/gipfel-throughput-760m-4s-1n-2227705.log`
- NSYS report: `/iopsstor/scratch/cscs/course_00282/gipfelsturm/nsys/gipfel-throughput-760m-4s-1n-rank0.nsys-rep`
- SQLite export: `/iopsstor/scratch/cscs/course_00282/gipfelsturm/nsys/gipfel-throughput-760m-4s-1n-rank0.sqlite`
- CSV summaries: `profile_stats/gipfel-throughput-760m-4s-1n_*.csv`

Throughput:
- Iteration 1: `9482` tokens/sec/GPU
- Iteration 2: `9784` tokens/sec/GPU
- Iteration 3: `10655` tokens/sec/GPU
- Iteration 4: `11266` tokens/sec/GPU

GPU kernel time, grouped from `cuda_gpu_kern_sum`:

| Group | Total kernel time | Share |
|---|---:|---:|
| NCCL collectives | 61.276 s | 53.9% |
| TransformerEngine / GEMM kernels | 27.045 s | 23.8% |
| CUDNN SDPA / flash attention | 13.730 s | 12.1% |
| Other Triton kernels | 5.846 s | 5.1% |
| Other kernels | 3.674 s | 3.2% |
| Norm kernels | 2.029 s | 1.8% |

Top individual GPU kernels:

| Share | Total | Kernel |
|---:|---:|---|
| 32.7% | 37.202 s | `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` |
| 11.8% | 13.417 s | `ncclDevKernel_AllGather_RING_LL` |
| 9.2% | 10.429 s | CUDNN SDPA flash backward |
| 9.0% | 10.259 s | `ncclDevKernel_AllReduce_Sum_f32_RING_LL` |
| 6.2% | 7.060 s | TransformerEngine GEMM (`nvjet_sm90_*`) |
| 5.2% | 5.917 s | TransformerEngine GEMM (`nvjet_sm90_*`) |

Initial interpretation:

- This 760m single-node setup is dominated by distributed optimizer / data-parallel communication, not attention. NCCL reduce-scatter, all-gather, and all-reduce account for about 54% of summed GPU kernel time in the short trace.
- Attention is already using CUDNN-generated flash SDPA kernels and is only about 12% of summed GPU kernel time. Attention backend swaps may still help, but this trace does not suggest attention is the largest immediate bottleneck for Thomas's lane.
- GEMM/TransformerEngine kernels are the second largest bucket at about 24%. This makes compile/fusion experiments plausible, but they need to be compared against communication-heavy overhead.
- CUDA API time is dominated by `cudaStreamSynchronize` at 66.9% of API time. Treat this as a symptom of synchronization/waiting rather than a standalone optimization target.
- The trace is short and includes warmup. Use it for direction, not final claims. A longer normal-partition trace after the run is stable would be better for final plots.

Next hypotheses:

1. Test whether `--use-distributed-optimizer` is helping or hurting at 1 node / 4 GPUs for the 760m throughput case.
2. Compare micro-batch/global-batch settings to see whether larger per-GPU work reduces the communication share.
3. For `torch.compile`, start with a tightly scoped experiment and watch for graph breaks around TransformerEngine and distributed optimizer boundaries.

## 2026-05-14: Distributed Optimizer Toggle Debug Test

Launcher change:
- Added `USE_DISTRIBUTED_OPTIMIZER`, defaulting to `1`.
- When set to `0`, the generated run omits `--use-distributed-optimizer`, `--overlap-grad-reduce`, and `--overlap-param-gather`.
- Megatron also requires `--use-precision-aware-optimizer` and `--main-grads-dtype bf16` to be disabled when the distributed optimizer is disabled, so those flags are gated by the same switch.

Runs:
- Job `2228691` failed before training because the first toggle version still passed `--use-precision-aware-optimizer`; Megatron asserted that this is only supported with distributed optimizer.
- Job `2228731` completed with `PARTITION=debug TIME_OVERRIDE=00:15:00 USE_DISTRIBUTED_OPTIMIZER=0 ./launch.sh throughput 760m 8 1`.
- Log: `logs/gipfel-throughput-760m-8s-1n-2228731.log`
- Final iteration: `20396` tokens/sec/GPU.
- Mid-run iterations were noisy: iteration 4 `15961`, iteration 5 `15661`, iteration 6 `17253`, iteration 7 `14197` tokens/sec/GPU.

Interpretation:
- The no-distributed-optimizer path is viable for 760m / 1 node / 4 GPUs once precision-aware optimizer flags are also disabled.
- Compared with the recent debug baseline final iteration of `11265` tokens/sec/GPU, this looks promising, but the debug partition and short run show enough variance that this should be treated only as a sanity signal.
- Next comparable evidence should be a normal-partition run or a longer controlled debug run, not started without agreement.

## 2026-05-14: Normal-Partition Distributed Optimizer Comparison

Commands:
- `PARTITION=normal TIME_OVERRIDE=00:30:00 USE_DISTRIBUTED_OPTIMIZER=0 ./launch.sh throughput 760m 20 1`
- `PARTITION=normal TIME_OVERRIDE=00:30:00 USE_DISTRIBUTED_OPTIMIZER=1 ./launch.sh throughput 760m 20 1`

Runs:
- No distributed optimizer: job `2229470`, log `logs/gipfel-throughput-760m-20s-1n-2229470.log`, completed in `4m28s`.
- Distributed optimizer baseline: job `2229668`, log `logs/gipfel-throughput-760m-20s-1n-2229668.log`, completed in `4m20s`.

Throughput:

| Run | Final iter | Avg iter 3-20 | Avg iter 10-20 |
|---|---:|---:|---:|
| `USE_DISTRIBUTED_OPTIMIZER=0` | `27611` | `32054` | `32128` |
| `USE_DISTRIBUTED_OPTIMIZER=1` | `27611` | `32737` | `32136` |

Interpretation:
- The debug-partition improvement did not reproduce under the controlled normal-partition comparison.
- Stable-window throughput is effectively tied, and the baseline is slightly higher on iter 3-20.
- Keep `USE_DISTRIBUTED_OPTIMIZER=1` as the default. Do not spend an NSYS pass on the no-distributed-optimizer variant unless later evidence changes this.
- Next practical lever is an MBS/GBS sweep or a tightly scoped compile experiment, not changing the distributed optimizer default.

## 2026-05-14: MBS Sweep Around 760m Default

Commands:
- `PARTITION=normal TIME_OVERRIDE=00:30:00 MBS_OVERRIDE=8 GBS_OVERRIDE=256 ./launch.sh throughput 760m 20 1`
- `PARTITION=normal TIME_OVERRIDE=00:30:00 MBS_OVERRIDE=2 GBS_OVERRIDE=256 ./launch.sh throughput 760m 20 1`

Runs:
- MBS 8: job `2231926`, log `logs/gipfel-throughput-760m-20s-1n-2231926.log`, completed in `8m32s`.
- MBS 2: job `2231946`, log `logs/gipfel-throughput-760m-20s-1n-2231946.log`, completed in `7m19s`.

Throughput compared with the MBS 4 normal baseline from job `2229668`:

| MBS | Final iter | Avg iter 3-20 | Avg iter 10-20 | Max allocated after warmup |
|---:|---:|---:|---:|---:|
| 2 | `14747` | `15051` | `16096` | ~17 GB |
| 4 | `27611` | `32737` | `32136` | ~30 GB |
| 8 | `15610` | `13120` | `14510` | ~54 GB |

Interpretation:
- The existing 760m default MBS 4 is clearly better than MBS 2 or MBS 8 for this 1-node normal-partition setup.
- Larger MBS 8 fits memory but roughly halves throughput, likely because the larger microbatch makes each microstep much slower without improving total tokens per iteration.
- Smaller MBS 2 also underperforms, likely from more microsteps per global batch and worse amortization.
- Keep MBS 4 for the current 760m throughput baseline.
- No existing Megatron launcher flag for full-model `torch.compile` was found; a compile experiment would require a deliberate runtime patch/hook rather than just adding a command-line argument.

## 2026-05-14: Longer Normal 760m Baseline

Command:
- `PARTITION=normal TIME_OVERRIDE=00:45:00 ./launch.sh throughput 760m 50 1`

Run:
- Job `2235436`, log `logs/gipfel-throughput-760m-50s-1n-2235436.log`, completed in `7m57s`.
- Final iteration: `37375` tokens/sec/GPU.
- Average iter 3-50: `42228` tokens/sec/GPU.
- Average iter 10-50: `40736` tokens/sec/GPU.
- Average iter 20-50: `41893` tokens/sec/GPU, median `41965`.

Interpretation:
- This is the cleanest 760m / 1-node normal baseline so far for final plots, but normal-partition throughput is very bursty on this run.
- Use stable-window averages and medians for plots instead of only the final iteration.

## 2026-05-15: Normal NSYS Profile Attempt Timed Out

Command:
- `PARTITION=normal TIME_OVERRIDE=00:45:00 ./profile_nsys.sh 760m 8 1`

Run:
- Job `2235710`, log `logs/gipfel-throughput-760m-8s-1n-2235710.log`.
- Slurm state: `TIMEOUT` after `45m15s`.
- Training reached all 8 iterations before timeout.
- Final iteration: `28336` tokens/sec/GPU.
- Iteration tokens/sec/GPU: 1 `16544`, 2 `19752`, 3 `19950`, 4 `25198`, 5 `19697`, 6 `25394`, 7 `29065`, 8 `28336`.

Profile artifact:
- Expected report path: `/iopsstor/scratch/cscs/course_00282/gipfelsturm/nsys/gipfel-throughput-760m-8s-1n-rank0.nsys-rep`.
- The report file exists but is `0` bytes because the job hit the Slurm time limit during NSYS report generation.
- No usable SQLite export or profile stats were produced from this attempt.

Interpretation:
- Normal-partition NSYS needs a larger wall-clock limit or fewer traced domains/steps; 8 training steps fit, but report finalization did not fit in `00:45:00`.
- Do not use this profile attempt for bottleneck claims. The earlier debug NSYS profile remains the only usable profile summary so far.
