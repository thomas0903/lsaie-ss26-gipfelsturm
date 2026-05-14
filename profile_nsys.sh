#!/bin/bash
#
# Submit a short throughput run under NVIDIA Nsight Systems.
#
# Usage:
#   ./profile_nsys.sh [model_size] [steps] [nodes]
#
# Defaults are intentionally small so this is suitable for first-pass
# bottleneck triage on the debug partition.

set -euo pipefail

MODEL_SIZE=${1:-760m}
STEPS=${2:-20}
NODES=${3:-1}

export PROFILE_NSYS=1
export PARTITION=${PARTITION:-debug}
export TIME_OVERRIDE=${TIME_OVERRIDE:-00:15:00}
export NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx,osrt,cublas,cudnn,nccl}

exec ./launch.sh throughput "$MODEL_SIZE" "$STEPS" "$NODES"
