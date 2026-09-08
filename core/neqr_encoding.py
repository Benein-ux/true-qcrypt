"""
NEQR (Novel Enhanced Quantum Representation) Encoder
=====================================================
Implements the NEQR model for representing classical grayscale images as
quantum states within a Qiskit QuantumCircuit.

Mathematical Basis
------------------
For an n×n image (n = 2^q), NEQR encodes each pixel f(i,j) ∈ [0, 255] as:

    |I⟩ = (1/n) Σ_{i,j} |f(i,j)⟩|ij⟩

where:
  - |f(i,j)⟩  is the 8-qubit grayscale value (color register)
  - |ij⟩       is the 2q-qubit position register (coordinate register)
  - The full state is a superposition over all pixel positions.

Register Layout (total qubits = 2q + 8):
  qubits [0 … 7]       → color register (MSB at qubit 0)
  qubits [8 … 8+2q-1]  → position register (row bits then col bits)
"""

import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class NEQREncoder:
    """
    Encodes a grayscale 2D NumPy image array into a Qiskit QuantumCircuit
    using the NEQR representation scheme.

    Parameters
    ----------
    image : np.ndarray
        2-D uint8 grayscale image. Must be square with side length being a
        power of 2.  Maximum supported side length: 16 (requires 40 qubits).

    Attributes
    ----------
    n : int
        Image side length (pixels).
    q : int
        Number of qubits required to index one axis (q = log2(n)).
    n_pos : int
        Total position register width = 2 * q (row qubits + col qubits).
    n_color : int
        Color register width = 8 (grayscale values 0–255).
    n_total : int
        Total qubit count = n_color + n_pos.
    circuit : QuantumCircuit
        The constructed NEQR circuit (populated by `encode()`).
    """

    COLOR_BITS: int = 8  # 8 qubits represent pixel intensity 0-255

    def __init__(self, image: np.ndarray) -> None:
        if image.ndim != 2:
            raise ValueError("NEQR encoder requires a 2-D grayscale image array.")
        h, w = image.shape
        if h != w:
            raise ValueError(f"Image must be square; got {h}×{w}.")
        if not (h > 0 and (h & (h - 1)) == 0):
            raise ValueError(f"Image side length must be a power of 2; got {h}.")
        if h > 64:
            raise ValueError(
                f"Image side length {h} exceeds the 64-pixel safety limit. "
                "Downscale before encoding."
            )

        self.image = image.astype(np.uint8)
        self.n = h
        self.q = int(math.log2(h))          # qubits per axis
        self.n_pos = 2 * self.q             # total position qubits
        self.n_color = self.COLOR_BITS
        self.n_total = self.n_color + self.n_pos

        # Quantum and classical registers
        self._color_reg = QuantumRegister(self.n_color, name="color")
        self._pos_reg = QuantumRegister(self.n_pos, name="pos")
        self._creg = ClassicalRegister(self.n_total, name="c")

        self.circuit: QuantumCircuit | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self) -> QuantumCircuit:
        """
        Build and return the NEQR quantum circuit encoding `self.image`.

        Steps
        -----
        1. Create a uniform superposition over all 2^(2q) position states
           using Hadamard gates on every position qubit.
        2. For each pixel (i, j), conditionally load the 8-bit grayscale
           value f(i,j) into the color register using multi-controlled X
           (MCX) gates that fire only when the position register holds |ij⟩.

        Returns
        -------
        QuantumCircuit
            The fully encoded NEQR circuit (without measurement).
        """
        qc = QuantumCircuit(self._color_reg, self._pos_reg, self._creg)

        # Step 1: Uniform superposition over all position states
        # H applied to every position qubit: |0⟩^{2q} → (1/√N)|00…0⟩+…+|11…1⟩
        for qb in self._pos_reg:
            qc.h(qb)

        qc.barrier(label="NEQR-superposition")

        # Step 2: Pixel-conditional color loading
        # For pixel at row i, col j with value v = f(i,j):
        #   Encode address: position register encodes |row_bits⟩|col_bits⟩
        #   For each bit k in v that is 1, apply MCX on color[k] controlled
        #   by the full position register (with X flips for 0-bits in address).
        n_pixels = self.n * self.n
        for flat_idx in range(n_pixels):
            row = flat_idx // self.n
            col = flat_idx % self.n

            pixel_val: int = int(self.image[row, col])
            if pixel_val == 0:
                # |0⟩ color register is already the default state — skip
                continue

            # Convert the flat address to a binary string of length n_pos
            addr_bits = format(flat_idx, f"0{self.n_pos}b")  # MSB first

            # Flip position qubits where address bit is '0' so that the
            # MCX fires on the correct basis state.
            zero_positions = [
                self._pos_reg[k]
                for k, bit in enumerate(addr_bits)
                if bit == "0"
            ]
            if zero_positions:
                qc.x(zero_positions)

            # For each bit in pixel_val that is '1', apply MCX targeting
            # the corresponding color qubit.
            color_bits = format(pixel_val, f"0{self.n_color}b")  # MSB first
            controls = list(self._pos_reg)
            for cb, bit in enumerate(color_bits):
                if bit == "1":
                    target = self._color_reg[cb]
                    qc.mcx(controls, target)

            # Un-flip the zero-bit position qubits to restore the superposition.
            if zero_positions:
                qc.x(zero_positions)

        qc.barrier(label="NEQR-encoded")
        self.circuit = qc
        return qc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_register_info(self) -> dict:
        """Return a summary of register allocation."""
        return {
            "image_size": f"{self.n}×{self.n}",
            "q (bits per axis)": self.q,
            "position_qubits": self.n_pos,
            "color_qubits": self.n_color,
            "total_qubits": self.n_total,
        }
