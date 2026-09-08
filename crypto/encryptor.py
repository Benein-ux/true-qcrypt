"""
Quantum Encryptor
=================
Orchestrates the monolithic NEQR quantum image encryption pipeline for both
grayscale and RGB images.

Pipeline (per channel)
----------------------
    Classical Image Channel (grayscale or R/G/B)
        ↓  Downscale to target_size × target_size (power of 2, ≤ 32)
    NEQR Encoding  →  QuantumCircuit |I⟩
        ↓  Quantum Arnold Cat Map  (SWAP/CSWAP gates on position register)
    Confusion Layer  →  position qubits permuted
        ↓  Deterministic CNOT Diffusion  (key_bits derived from key_seed)
    Diffusion Layer  →  color register altered
        ↓  QASM Measurement (Aer simulator, dominant-shot pixel extraction)
    Encrypted Channel Array  (uint8 NumPy array)

RGB Support
-----------
When color_mode='rgb', the pipeline is applied independently to the R, G,
and B channels using the same key_seed (so all three can be decrypted with
the same key).  The three encrypted channels are stacked into a (H,W,3) array.

Academic Compliance
-------------------
- Monolithic NEQR circuit: no block-splitting, no chunking.
- Supported sizes: 4×4, 8×8, 16×16, 32×32.
- All encryption logic runs inside QuantumCircuit objects.
- key_seed and key_bits are stored on the encryptor instance for
  handoff to QuantumDecryptor.
"""

import time
import math
import numpy as np
from PIL import Image

from core.neqr_encoding import NEQREncoder
from core.quantum_gates import QuantumImageGates
from core.measurement import QuantumMeasurement

# Supported monolithic NEQR sizes
VALID_SIZES: tuple[int, ...] = (4, 8, 16, 32, 64)
MAX_IMAGE_SIZE: int = 64

# ── LSQB Steganography (Quantum Watermarking) ─────────────────────────
# Author identity string embedded into the LSB (q0 color qubit) of every pixel.
WATERMARK_ID: str = "IS-2301020531"

def _generate_watermark_bits(n_pixels: int) -> list[int]:
    """
    Convert WATERMARK_ID to an 8-bit ASCII binary stream, tiled to fill
    exactly ``n_pixels`` positions.

    Returns a list of int (0 or 1), length == n_pixels.
    """
    base_bits: list[int] = []
    for ch in WATERMARK_ID:
        base_bits.extend(int(b) for b in format(ord(ch), "08b"))
    period = len(base_bits)
    return [base_bits[i % period] for i in range(n_pixels)]


def embed_lsqb_watermark(channel: np.ndarray) -> np.ndarray:
    """
    Overwrite the Least Significant Bit of every pixel in *channel* with the
    corresponding bit from the WATERMARK_ID bit-stream.

    Parameters
    ----------
    channel : np.ndarray  shape (N, N)  dtype uint8
        Grayscale channel BEFORE NEQR encoding.

    Returns
    -------
    np.ndarray  same shape and dtype — identical to input except LSBs replaced.

    Implementation note
    -------------------
    This is purely classical preprocessing.  The modified array is passed
    into NEQREncoder unchanged, so the q0 color qubit encodes the watermark
    bit directly.  Because the confusion and diffusion layers are unitary
    and fully reversible, the statevector decryption phase restores the
    original LSBs — including the embedded watermark — with SSIM = 1.0.
    """
    flat = channel.flatten().astype(np.uint8)
    bits = _generate_watermark_bits(len(flat))
    # Clear LSB then OR with watermark bit: (pixel & 0xFE) | bit
    watermarked = (flat & np.uint8(0xFE)) | np.array(bits, dtype=np.uint8)
    return watermarked.reshape(channel.shape)


def extract_lsqb_watermark(channel: np.ndarray, expected: str = WATERMARK_ID) -> dict:
    """
    Extract the LSQB watermark from the LSBs of a decrypted channel.

    Parameters
    ----------
    channel : np.ndarray  shape (N, N)  dtype uint8
        Decrypted grayscale channel.
    expected : str
        The expected watermark string for verification.

    Returns
    -------
    dict with keys:
        'raw_bits'    : list[int]     — extracted LSBs
        'decoded_str' : str           — full decoded text (repeating)
        'signature'   : str           — first len(expected) chars decoded
        'verified'    : bool          — True if signature == expected
    """
    flat = channel.flatten().astype(np.uint8)
    bits = [int(p) & 1 for p in flat]

    # Decode every complete 8-bit group into a character
    chars: list[str] = []
    for i in range(0, len(bits) - 7, 8):
        byte_val = int("".join(str(b) for b in bits[i : i + 8]), 2)
        chars.append(chr(byte_val) if 32 <= byte_val <= 126 else "?")
    decoded = "".join(chars)

    # The image may not hold all len(expected) chars (e.g. 8×8 → 8 chars).
    # Compare against the prefix of expected that actually fits.
    capacity = len(chars)                         # how many chars we decoded
    sig_len  = min(len(expected), capacity)       # how many to compare
    signature     = decoded[:sig_len]
    expected_part = expected[:sig_len]
    verified      = (signature == expected_part) and sig_len > 0

    return {
        "raw_bits":    bits,
        "decoded_str": decoded,
        "signature":   signature,
        "verified":    verified,
        "capacity_chars": capacity,
    }



class QuantumEncryptor:
    """
    Full monolithic quantum image encryptor supporting grayscale and RGB.

    Parameters
    ----------
    image : PIL.Image.Image | np.ndarray
        Source image (any PIL mode).
    target_size : int, optional
        NEQR circuit side length — must be 4, 8, 16, or 32 (default 16).
    cat_iterations : int, optional
        Arnold Cat Map repetitions (default 3).
    shots : int, optional
        Base QASM simulator shots (auto-scaled for coverage, default 8192).
    key_seed : int, optional
        Seed for the deterministic diffusion key PRNG (default 42).
        **Must match the value given to QuantumDecryptor for lossless recovery.**
    color_mode : str, optional
        'grayscale' (default) or 'rgb'.  RGB runs the pipeline on each channel.
    """

    def __init__(
        self,
        image,
        target_size: int = 16,
        cat_iterations: int = 3,
        shots: int = 8192,
        key_seed: int = 42,
        color_mode: str = "grayscale",
        use_statevector: bool = True,
        embed_watermark: bool = False,
    ) -> None:
        # Clamp to nearest valid size
        target_size = self._clamp_size(target_size)
        self.target_size = target_size
        self.cat_iterations = cat_iterations
        self.shots = shots
        self.key_seed = key_seed
        self.color_mode = color_mode.lower()
        self.use_statevector = use_statevector
        self.embed_watermark = embed_watermark

        # Preprocess and store the source channels
        self._pil_source = self._to_pil(image)
        self.original_channels: dict[str, np.ndarray] = self._preprocess()

        # Outputs populated by encrypt()
        self.encrypted_channels: dict[str, np.ndarray] = {}
        self.key_bits: list[int] = []          # derived from key_seed, stored for decryptor
        self.circuit = None                    # last channel's circuit (for visualisation)
        self.elapsed_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encrypt(self, progress_cb=None) -> np.ndarray:
        """
        Run the full quantum encryption pipeline.

        Parameters
        ----------
        progress_cb : callable(channel_name: str, fraction: float) | None
            Optional callback invoked after each channel completes.
            Fraction is in [0, 1].

        Returns
        -------
        np.ndarray
            Grayscale: shape (H, W),     dtype uint8.
            RGB:       shape (H, W, 3),  dtype uint8.
        """
        t_start = time.perf_counter()
        channel_names = list(self.original_channels.keys())
        n_channels = len(channel_names)

        for idx, ch_name in enumerate(channel_names):
            # Apply watermark to the stored original channel BEFORE encryption
            # so that original_image reflects the watermarked pixels and the
            # combined-circuit decryptor naturally restores them.
            if self.embed_watermark:
                self.original_channels[ch_name] = embed_lsqb_watermark(
                    self.original_channels[ch_name]
                )
            enc_ch, qc, key_bits = self._encrypt_channel(
                self.original_channels[ch_name]
            )
            self.encrypted_channels[ch_name] = enc_ch
            self.circuit = qc        # keep last for circuit visualisation
            self.key_bits = key_bits # same seed → same key_bits every channel

            if progress_cb is not None:
                progress_cb(ch_name, (idx + 1) / n_channels)

        self.elapsed_time = time.perf_counter() - t_start

        return self._assemble_output(self.encrypted_channels)

    @property
    def original_image(self) -> np.ndarray:
        """Return the preprocessed original image (grayscale or RGB)."""
        return self._assemble_output(self.original_channels)

    def get_circuit(self):
        """Return the last constructed QuantumCircuit (after encrypt())."""
        if self.circuit is None:
            raise RuntimeError("Call encrypt() first.")
        return self.circuit

    def get_circuit_stats(self) -> dict:
        """Gate counts and timing for the last-built circuit."""
        if self.circuit is None:
            raise RuntimeError("Call encrypt() first.")
        return {
            "num_qubits": self.circuit.num_qubits,
            "depth": self.circuit.depth(),
            "gate_counts": dict(self.circuit.count_ops()),
            "elapsed_seconds": round(self.elapsed_time, 2),
            "color_mode": self.color_mode,
            "target_size": self.target_size,
        }

    # ------------------------------------------------------------------
    # Internal: single-channel encrypt
    # ------------------------------------------------------------------

    def _encrypt_channel(
        self, channel: np.ndarray
    ) -> tuple[np.ndarray, object, list]:
        """
        Run the NEQR → confusion → diffusion → measurement pipeline on one
        grayscale channel array.

        The LSQB watermark (if enabled) has already been applied to *channel*
        by the ``encrypt()`` caller before this method is invoked.

        Returns (encrypted_array, circuit, key_bits).
        """
        # 1. NEQR encode (channel already has watermark LSBs if enabled)
        encoder = NEQREncoder(channel)
        qc = encoder.encode()
        n_color = encoder.n_color
        n_pos = encoder.n_pos

        # 2. Confusion (Arnold Cat Map) + Diffusion (deterministic CNOT key)
        gates = QuantumImageGates(
            qc, n_color, n_pos,
            iterations=self.cat_iterations,
            key_seed=self.key_seed,
        )
        gates.apply_confusion()
        gates.apply_diffusion()

        # 3. Measure and reconstruct
        measurer = QuantumMeasurement(
            qc, n_color, n_pos, self.target_size, self.shots,
            use_statevector=self.use_statevector,
        )
        encrypted = measurer.measure_and_reconstruct()

        return encrypted, qc, gates.key_bits



    # ------------------------------------------------------------------
    # Internal: preprocessing & assembly
    # ------------------------------------------------------------------

    def _preprocess(self) -> dict[str, np.ndarray]:
        """
        Convert the source PIL image to the required channel dictionary.

        Returns
        -------
        dict with key 'L' for grayscale, or keys 'R','G','B' for RGB.
        Each value is a (target_size × target_size) uint8 numpy array.
        """
        sz = self.target_size
        if self.color_mode == "rgb":
            rgb = self._pil_source.convert("RGB").resize((sz, sz), Image.LANCZOS)
            r, g, b = np.array(rgb).transpose(2, 0, 1)   # each shape (sz,sz)
            return {"R": r.astype(np.uint8),
                    "G": g.astype(np.uint8),
                    "B": b.astype(np.uint8)}
        else:
            gray = self._pil_source.convert("L").resize((sz, sz), Image.LANCZOS)
            return {"L": np.array(gray, dtype=np.uint8)}

    def _assemble_output(self, channels: dict) -> np.ndarray:
        """Stack channel dict back into a grayscale or RGB numpy array."""
        if self.color_mode == "rgb":
            return np.dstack([channels["R"], channels["G"], channels["B"]])
        return channels["L"]

    @staticmethod
    def _clamp_size(size: int) -> int:
        """Return the nearest valid NEQR size ≤ size, min 4."""
        for s in reversed(VALID_SIZES):
            if size >= s:
                return s
        return VALID_SIZES[0]

    @staticmethod
    def _to_pil(image) -> Image.Image:
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        if isinstance(image, Image.Image):
            return image
        raise TypeError(f"Unsupported image type: {type(image)}")
