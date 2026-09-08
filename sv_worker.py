"""
sv_worker.py — Statevector simulation worker (subprocess-safe)
================================================================
This script is invoked as a STANDALONE SUBPROCESS by core/measurement.py.
It MUST NOT be imported as a package module — it exists as a plain script
so that native libraries load in a fresh OS process with no inherited
thread state from Streamlit.

Protocol
--------
  stdin  → pickle.dumps(QuantumCircuit)
  stdout → pickle.dumps(np.ndarray complex128)  [the statevector]
  stderr → error messages

Why subprocess instead of AerSimulator in-process
--------------------------------------------------
AerSimulator's C++ backend initialises an OpenMP thread-pool when first
called.  Streamlit's uvicorn server creates its own OS threads for async
I/O during startup.  When AerSimulator tries to spawn OpenMP threads from
within one of Streamlit's threads (or from the main thread while Streamlit's
threads are active), the OpenMP runtime detects a fork-safety violation and
raises SIGSEGV.

Running AerSimulator in a completely isolated subprocess (spawned AFTER
Streamlit's threads start) avoids the collision entirely:  the child process
has no Streamlit threads, no OpenMP state to conflict with, and sets its own
OMP_NUM_THREADS=1 before any native library is loaded.
"""

# ── 1. Set thread env-vars BEFORE any native library is imported ───────
import os
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
os.environ["QISKIT_PARALLEL"]        = "FALSE"

# ── 2. Now safe to import native packages ──────────────────────────────
import sys
import pickle
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile

# ── 3. Read pickled circuit from stdin ────────────────────────────────
try:
    circuit_bytes = sys.stdin.buffer.read()
    qc = pickle.loads(circuit_bytes)
except Exception as e:
    sys.stderr.write(f"sv_worker: failed to deserialise circuit: {e}\n")
    sys.exit(1)

# ── 4. Run statevector simulation ──────────────────────────────────────
try:
    qc.save_statevector()
    sim = AerSimulator(
        method="statevector",
        max_parallel_threads=1,      # no new OS threads
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )
    result  = sim.run(transpile(qc, sim)).result()
    sv      = np.asarray(result.get_statevector())
except Exception as e:
    sys.stderr.write(f"sv_worker: simulation error: {e}\n")
    sys.exit(2)

# ── 5. Write pickled statevector to stdout ────────────────────────────
try:
    sys.stdout.buffer.write(pickle.dumps(sv))
except Exception as e:
    sys.stderr.write(f"sv_worker: failed to serialise result: {e}\n")
    sys.exit(3)
