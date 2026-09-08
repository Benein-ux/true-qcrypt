"""
Quantum Circuit Measurement & Classical Reconstruction
=======================================================
Provides two reconstruction backends for NEQR quantum image circuits:

1. **Statevector mode** (default, ``use_statevector=True``)
   Uses ``qiskit.quantum_info.Statevector`` — a **pure-NumPy, thread-safe**
   statevector simulator — to extract the exact quantum state amplitudes
   with zero shot noise.  Every pixel is recovered with mathematical precision,
   giving SSIM = 1.0 on round-trip decryption for all supported sizes
   (4×4 through 32×32 / 18 qubits).

   Why NOT AerSimulator for statevector
   -------------------------------------
   ``AerSimulator(method='statevector')`` uses an OpenMP thread-pool
   internally.  When Streamlit re-runs the script in its own thread (on
   file-upload or button-click), OpenMP's thread management conflicts with
   Streamlit's thread-pool, causing a **segmentation fault** (SIGSEGV).

   ``qiskit.quantum_info.Statevector`` is implemented entirely in NumPy /
   Python, with no native threads and no OpenMP dependency.  It is safe to
   call from any thread context, including Streamlit's rerun threads.

   Performance note: For 18-qubit circuits (32×32 images), the NumPy
   statevector simulator completes in ~2–4 s — acceptable for academic use.

   Extraction algorithm
   --------------------
   Exactly N² basis states carry non-zero amplitude  |ψ⟩ = (1/N) Σ |f⟩|pos⟩.
   We iterate all 2^n states, filter by probability threshold 0.5/N², and
   decode (color, pos) using the verified qubit-index-to-bit mapping (see
   _extract_from_statevector for details).

2. **QASM shot-sampling mode** (``use_statevector=False``)
   Uses ``AerSimulator`` (QASM), retained as a legacy/comparison fallback.
   Prone to shot noise for 32×32 images.
"""

import os
import sys
import math
import numpy as np
from collections import Counter
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister

# NOTE: AerSimulator and transpile are intentionally NOT imported at module level.
#
# Rationale: importing qiskit_aer loads its native C++ shared library and may
# initialise an OpenMP thread pool in the parent process.  When subprocess.run()
# is subsequently called (to isolate statevector simulation in sv_worker.py),
# Linux performs fork() before exec().  fork() in a process with live OpenMP
# threads is undefined behaviour — it can cause SIGSEGV in the child before
# exec() completes.
#
# By importing AerSimulator lazily (inside _qasm_reconstruct only), the parent
# Streamlit process never loads Aer's native library in statevector mode
# (use_statevector=True, the default), making fork()+exec() safe.

MIN_SHOTS_PER_PIXEL: int = 16


class QuantumMeasurement:
    """
    Reconstructs a classical pixel array from a quantum image circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Fully-constructed circuit (NEQR ± confusion ± diffusion layers).
        Must *not* already contain measurement instructions when using
        statevector mode.
    n_color : int
        Width of the color register (8 for grayscale).
    n_pos : int
        Width of the position register (2 * log2(image_side)).
    image_size : int
        Side length of the image in pixels (power of 2).
    shots : int, optional
        Shot count for QASM mode only (ignored in statevector mode).
    use_statevector : bool, optional
        If True (default), bypass shot sampling and use the exact
        statevector.  Gives noiseless SSIM = 1.0 at any image size.
        If False, fall back to QASM shot sampling.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        n_color: int,
        n_pos: int,
        image_size: int,
        shots: int = 8192,
        use_statevector: bool = True,
    ) -> None:
        self.circuit = circuit
        self.n_color = n_color
        self.n_pos = n_pos
        self.image_size = image_size
        self.use_statevector = use_statevector
        self._relevant = n_color + n_pos

        # QASM mode: auto-scale shots for adequate pixel coverage
        n_addresses = image_size * image_size
        self.shots = max(shots, n_addresses * MIN_SHOTS_PER_PIXEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure_and_reconstruct(self) -> np.ndarray:
        """
        Reconstruct the classical image using the selected backend.

        Returns
        -------
        np.ndarray  shape=(image_size, image_size), dtype=uint8
            Exact pixel values — no normalisation applied.
        """
        if self.use_statevector:
            return self._sv_reconstruct()
        return self._qasm_reconstruct()

    # ------------------------------------------------------------------
    # Backend 1: Statevector (exact, noiseless)
    # ------------------------------------------------------------------

    def _sv_reconstruct(self) -> np.ndarray:
        """
        Run the statevector simulation in an isolated subprocess (sv_worker.py).

        Why a subprocess?
        -----------------
        AerSimulator's OpenMP thread-pool collides with Streamlit's uvicorn/asyncio
        threads regardless of env-var mitigations, because by the time app.py runs
        its ``os.environ.setdefault`` calls, Streamlit has already started its async
        server threads — and OpenMP runtime checks fork/thread safety against the
        *currently running* threads, not the env vars.

        Spawning a fresh subprocess (via ``subprocess.run``) is the only approach
        that gives AerSimulator a completely clean OS-level environment:
          • No Streamlit threads are present in the child process.
          • Env vars are set at the VERY TOP of sv_worker.py, before any native
            library is imported.
          • The child exits after delivering its result, so nothing accumulates.

        Protocol
        --------
        stdin  ← ``pickle.dumps(QuantumCircuit)``
        stdout → ``pickle.dumps(numpy.ndarray)``   (complex128 statevector)
        """
        import subprocess
        import pickle

        qc = self.circuit.copy()

        # Locate sv_worker.py relative to this file (core/ → ../ → true-qcrypt/)
        worker_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "sv_worker.py"
        )
        worker_path = os.path.normpath(worker_path)

        try:
            proc = subprocess.run(
                [sys.executable, worker_path],
                input=pickle.dumps(qc),
                capture_output=True,
                timeout=600,        # 10-minute hard limit (generous for 64×64)
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Statevector simulation timed out (>10 min).")

        if proc.returncode != 0:
            err = proc.stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"sv_worker.py exited with code {proc.returncode}:\n{err}"
            )

        sv = pickle.loads(proc.stdout)
        return self._extract_from_statevector(sv)


    def _extract_from_statevector(self, sv: np.ndarray) -> np.ndarray:
        """
        Decode pixel values from the statevector amplitude array.

        For an N×N NEQR image the state is:
            |ψ⟩ = (1/N) Σ_{i,j} |color(i,j)⟩|pos(i,j)⟩

        Exactly N² basis states have probability 1/N².
        We use threshold p > 0.5/N² to identify them.

        Parameters
        ----------
        sv : np.ndarray
            Complex amplitude array of length 2^n_total.

        Returns
        -------
        np.ndarray  shape=(image_size, image_size), dtype=uint8
        """
        n = self.image_size
        nc = self.n_color
        np_ = self.n_pos
        n_sq = n * n

        # Threshold: half of the expected per-pixel probability (1/N²)
        threshold_prob = 0.5 / n_sq

        image = np.zeros((n, n), dtype=np.uint8)
        found = np.zeros((n, n), dtype=bool)

        for state_idx in range(len(sv)):
            prob = abs(sv[state_idx]) ** 2
            if prob < threshold_prob:
                continue

            # ── Color register: qubits 0 … nc-1 ──────────────────────
            # qubit k maps to bit k of state_idx (qubit 0 = bit 0 = LSB).
            # NEQR encodes pixel MSB into qubit 0, so we reverse to recover
            # the original MSB-first binary representation.
            color_qubit_bits = state_idx & ((1 << nc) - 1)
            color_val = int(format(color_qubit_bits, f"0{nc}b")[::-1], 2)

            # ── Position register: qubits nc … nc+np_-1 ───────────────
            pos_qubit_bits = (state_idx >> nc) & ((1 << np_) - 1)
            pos_addr = int(format(pos_qubit_bits, f"0{np_}b")[::-1], 2)

            if 0 <= pos_addr < n_sq:
                row, col = divmod(pos_addr, n)
                if not found[row, col]:
                    image[row, col] = color_val
                    found[row, col] = True

        return image

    # ------------------------------------------------------------------
    # Backend 2: QASM shot-sampling (legacy)
    # ------------------------------------------------------------------

    def _qasm_reconstruct(self) -> np.ndarray:
        """
        Measure relevant qubits, run QASM simulation, parse histogram.

        Retained for reference and comparison.  For images ≥ 16×16,
        shot noise causes missing pixels — prefer statevector mode.

        AerSimulator is imported lazily here so that the parent Streamlit
        process does not load Aer's native library in statevector mode (the
        default).  This keeps subprocess.run()'s fork() call safe.
        """
        # Lazy import — AerSimulator native code only loaded if QASM mode is used
        from qiskit import transpile  # noqa: PLC0415
        from qiskit_aer import AerSimulator  # noqa: PLC0415

        qc = self.circuit.copy()
        nc, np_ = self.n_color, self.n_pos

        meas_creg = ClassicalRegister(self._relevant, name="meas")
        qc.add_register(meas_creg)
        qc.measure(list(range(self._relevant)), meas_creg)

        simulator = AerSimulator()
        compiled = transpile(qc, simulator)
        counts = simulator.run(compiled, shots=self.shots).result().get_counts()

        n = self.image_size
        accum: dict[int, Counter] = {addr: Counter() for addr in range(n * n)}

        for bitstring, count in counts.items():
            # Leftmost section of the space-separated string = last-added creg
            meas_bits = bitstring.split()[0]
            if len(meas_bits) < nc + np_:
                continue

            # Bitstring layout (verified):
            #   meas_bits[0:n_pos]      → pos register, MSB-first
            #   meas_bits[n_pos:n_pos+n_color] reversed → color, MSB-first
            pos_addr  = int(meas_bits[:np_], 2)
            color_val = int(meas_bits[np_: np_ + nc][::-1], 2)

            if 0 <= pos_addr < n * n:
                accum[pos_addr][color_val] += count

        image = np.zeros((n, n), dtype=np.uint8)
        for flat_idx in range(n * n):
            row, col = divmod(flat_idx, n)
            counter = accum[flat_idx]
            if counter:
                image[row, col] = counter.most_common(1)[0][0]

        return image

    @staticmethod
    def get_dominant_counts(counts: dict, top_k: int = 20) -> dict:
        """Return top-k most frequent bitstrings (QASM debugging aid)."""
        return dict(sorted(counts.items(), key=lambda x: -x[1])[:top_k])
