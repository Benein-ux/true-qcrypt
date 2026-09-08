#!/usr/bin/env bash
# TrueQCrypt launcher — sets env vars to prevent Qiskit-Aer/OpenMP segfault
# Usage: ./run.sh [--server.port 8501]
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export QISKIT_PARALLEL=FALSE
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/streamlit run app.py "$@"
