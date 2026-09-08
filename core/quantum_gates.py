"""
Quantum Image Encryption Gates
================================
Provides two families of quantum operations for image encryption:

1. **Quantum Confusion (Scrambling)** — Quantum Arnold Cat Map
   Sequential SWAP/CSWAP gates on the position register scramble pixel
   locations in superposition.  The gate sequence is recorded so that the
   *exact* inverse can be replayed for lossless decryption.

2. **Quantum Diffusion (Value Alteration)** — Deterministic CNOT Diffusion
   A *deterministic* key bitstring (derived from a seeded PRNG or supplied
   explicitly) controls which color-register qubits are flipped via X gates
   *before* and *after* a CNOT layer.  Because the key is deterministic, the
   inverse operation is identical and perfectly reproducible.

   Previous design flaw: Using Hadamard-seeded key registers produced a
   *different* random superposition on every run, making decryption impossible.
   Fix: Replace Hadamard key with a deterministic key_bits sequence that is
   stored alongside the circuit for later reversal.

Gate Inversion Convention
--------------------------
Every SWAP is self-inverse (SWAP† = SWAP).
Every CNOT is self-inverse (CNOT† = CNOT).
X gates are self-inverse (X† = X).
The *order* of gate application must be reversed for the adjoint.

The cat-map gate sequence is stored in `self.cat_gate_log` and can be
replayed in reverse to implement U_confusion†.
"""

import math
import hashlib
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


class QuantumImageGates:
    """
    Appends deterministic quantum confusion and diffusion layers to an NEQR circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        NEQR-encoded circuit with named registers 'color' and 'pos'.
    n_color : int
        Width of color register (= 8 for grayscale).
    n_pos : int
        Width of position register (= 2 * log2(image_side)).
    iterations : int, optional
        Arnold Cat Map repetitions (default 3).
    key_seed : int | None, optional
        Seed for the deterministic diffusion key PRNG.  Must be the same
        value used during encryption to achieve lossless decryption.
        Defaults to 42.
    key_bits : list[int] | None, optional
        Explicit 8-bit key sequence (overrides key_seed).  If supplied, used
        directly as the deterministic diffusion pattern.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        n_color: int,
        n_pos: int,
        iterations: int = 3,
        key_seed: int = 42,
        key_bits: list | None = None,
    ) -> None:
        self.qc = circuit
        self.n_color = n_color
        self.n_pos = n_pos
        self.q = n_pos // 2
        self.iterations = iterations
        self.key_seed = key_seed

        # Deterministic key: 8 bits (one per color qubit) derived from seed
        if key_bits is not None:
            self.key_bits: list[int] = list(key_bits)[: n_color]
        else:
            rng = np.random.default_rng(key_seed)
            self.key_bits = [int(b) for b in rng.integers(0, 2, size=n_color)]

        # Ordered log of (gate_type, qubit_indices) for exact inversion
        self.cat_gate_log: list[tuple] = []

        self._color_reg = self._find_register("color")
        self._pos_reg = self._find_register("pos")

    # ------------------------------------------------------------------
    # Confusion: Quantum Arnold Cat Map
    # ------------------------------------------------------------------

    def apply_confusion(self) -> "QuantumImageGates":
        """
        Apply the Quantum Arnold Cat Map to the position register.

        Mathematical Action
        -------------------
        Classical Arnold Cat Map on (x, y) ∈ Z_n × Z_n:
            [x']   [1  1] [x]       [x + y  mod n ]
            [y'] = [1  2] [y]  →    [x + 2y mod n ]

        Quantum approximation via SWAP/CSWAP layers:
          Step A: row SWAPs forward   (cyclic shift on x bits)
          Step B: col SWAPs forward   (cyclic shift on y bits)
          Step C: CSWAP cross-coupling (implements x+y mixing term)
          Step D: row SWAPs backward  (adds 2y contribution)

        Gate log is recorded for exact adjoint reconstruction.

        Returns self (method chaining).
        """
        row_q = [self._pos_reg[k] for k in range(self.q)]
        col_q = [self._pos_reg[k + self.q] for k in range(self.q)]

        for _ in range(self.iterations):
            self.qc.barrier(label="cat-start")

            # Step A: row forward SWAPs
            for k in range(self.q - 1):
                self.qc.swap(row_q[k], row_q[k + 1])
                self.cat_gate_log.append(("swap", (row_q[k], row_q[k + 1])))

            # Step B: col forward SWAPs
            for k in range(self.q - 1):
                self.qc.swap(col_q[k], col_q[k + 1])
                self.cat_gate_log.append(("swap", (col_q[k], col_q[k + 1])))

            # Step C: CSWAP cross-coupling
            for k in range(self.q - 1):
                self.qc.cswap(row_q[k], col_q[k], col_q[k + 1])
                self.cat_gate_log.append(("cswap", (row_q[k], col_q[k], col_q[k + 1])))

            # Step D: row backward SWAPs
            for k in range(self.q - 2, -1, -1):
                self.qc.swap(row_q[k], row_q[k + 1])
                self.cat_gate_log.append(("swap", (row_q[k], row_q[k + 1])))

            self.qc.barrier(label="cat-end")

        return self

    def build_cat_gate_log(self) -> "QuantumImageGates":
        """
        Record the forward cat-map gate sequence into `self.cat_gate_log`
        WITHOUT appending any gates to the quantum circuit.

        This is used by the decryptor to populate the log so that
        `apply_inverse_confusion()` can replay it in reverse.  Calling
        `apply_confusion()` instead would incorrectly append forward gates
        to a circuit that already encodes the encrypted image.

        Returns self (method chaining).
        """
        row_q = [self._pos_reg[k] for k in range(self.q)]
        col_q = [self._pos_reg[k + self.q] for k in range(self.q)]

        for _ in range(self.iterations):
            for k in range(self.q - 1):
                self.cat_gate_log.append(("swap", (row_q[k], row_q[k + 1])))
            for k in range(self.q - 1):
                self.cat_gate_log.append(("swap", (col_q[k], col_q[k + 1])))
            for k in range(self.q - 1):
                self.cat_gate_log.append(("cswap", (row_q[k], col_q[k], col_q[k + 1])))
            for k in range(self.q - 2, -1, -1):
                self.cat_gate_log.append(("swap", (row_q[k], row_q[k + 1])))

        return self

    def apply_inverse_confusion(self) -> "QuantumImageGates":
        """
        Apply the exact inverse (adjoint) of the Arnold Cat Map gate sequence.

        Since SWAP† = SWAP and CSWAP† = CSWAP, the inverse is obtained by
        replaying the gate log in *reverse order*.  Call `build_cat_gate_log()`
        (decryptor path) or `apply_confusion()` (encryptor path) first to
        populate `self.cat_gate_log`.

        Returns self (method chaining).
        """
        self.qc.barrier(label="inv-cat-start")
        for gate_type, qubits in reversed(self.cat_gate_log):
            if gate_type == "swap":
                self.qc.swap(*qubits)
            elif gate_type == "cswap":
                self.qc.cswap(*qubits)
        self.qc.barrier(label="inv-cat-end")
        return self

    # ------------------------------------------------------------------
    # Diffusion: Deterministic CNOT-based diffusion
    # ------------------------------------------------------------------

    def apply_diffusion(self) -> "QuantumImageGates":
        """
        Apply deterministic quantum diffusion to the color register.

        Mathematical Action
        -------------------
        For each color qubit i:
          - If key_bits[i] == 1: apply X(color[i])  (flips the qubit)
          - Apply CNOT(color[i] → color[(i+1) % n_color])  (avalanche)
          - If key_bits[i] == 1: apply X(color[i])  (un-flip, net result is
            controlled bit-flip on the neighbor)

        Equivalently: for positions where key_bits[i]=1, the (i+1)-th color
        qubit is flipped.  The X-CNOT-X sandwich is a deterministic operation
        with a known, reproducible inverse.

        Key Design
        ----------
        The key_bits list is derived from a seeded PRNG (key_seed), making
        it perfectly reproducible across encryption and decryption sessions.

        Returns self (method chaining).
        """
        self.qc.barrier(label="diffusion-start")

        for i in range(self.n_color):
            # Conditionally flip color[i] based on key bit
            if self.key_bits[i] == 1:
                self.qc.x(self._color_reg[i])
            # Avalanche: propagate altered value to next color qubit
            next_i = (i + 1) % self.n_color
            self.qc.cx(self._color_reg[i], self._color_reg[next_i])

        self.qc.barrier(label="diffusion-end")
        return self

    def apply_inverse_diffusion(self) -> "QuantumImageGates":
        """
        Apply the exact inverse of the diffusion layer.

        Since CNOT† = CNOT and X† = X, the inverse is obtained by replaying
        the diffusion operations in *reverse order*:
            for i in reversed(range(n_color)):
                CNOT(color[i] → color[(i+1) % n_color])
                if key_bits[i] == 1: X(color[i])

        Returns self (method chaining).
        """
        self.qc.barrier(label="inv-diffusion-start")

        for i in reversed(range(self.n_color)):
            next_i = (i + 1) % self.n_color
            self.qc.cx(self._color_reg[i], self._color_reg[next_i])
            if self.key_bits[i] == 1:
                self.qc.x(self._color_reg[i])

        self.qc.barrier(label="inv-diffusion-end")
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_register(self, name: str):
        """Locate a named QuantumRegister within the circuit."""
        for reg in self.qc.qregs:
            if reg.name == name:
                return reg
        raise ValueError(
            f"Register '{name}' not found. "
            f"Available: {[r.name for r in self.qc.qregs]}"
        )
