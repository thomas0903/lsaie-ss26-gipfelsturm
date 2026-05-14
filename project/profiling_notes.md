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
