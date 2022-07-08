#!/usr/bin/env bash
# This wrapper script sets the number of threads in common parallelization
# libraries to 1. This avoids spawning 80 threads per process on DGX.
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 "$@"
