# TrueQCrypt 🔐⚛️

**A Monolithic Quantum Image Encryption & LSQB Steganography System** built with Qiskit and the NEQR (Novel Enhanced Quantum Representation) model.

---

## Overview

TrueQCrypt encodes digital images into pure quantum states and performs encryption and steganography entirely using reversible quantum gates inside `qiskit.QuantumCircuit` objects.

### Features
- **Monolithic NEQR Architecture**: Whole-image encoding into a single quantum register ($4\times4$ up to $64\times64$ pixels, up to 20 qubits) with zero classical chunking or block-splicing.
- **Grayscale & RGB Modes**: 3-channel classical quantum decomposition running independent NEQR circuits on Red, Green, and Blue channels.
- **Quantum Arnold Cat Map (Confusion)**: Scrambles pixel coordinates in spatial superposition using quantum SWAP and Fredkin (CSWAP) gates.
- **Deterministic CNOT Diffusion**: Flips color register amplitudes using key-derived controlled-NOT gates for high diffusion and entropy.
- **LSQB Quantum Steganography (Watermarking)**: Embeds an author identity string (`IS-2301020531`) into the least significant color qubit ($q_0$) with lossless extraction upon statevector decryption.
- **Noiseless Statevector Decryption**: Exact amplitude readout yielding mathematically perfect round-trip reconstruction ($\text{SSIM} = 1.0$).
- **Comprehensive Security Analysis**: Real-time evaluation of NPCR, UACI, Shannon Entropy, SSIM, and directional correlation coefficients (H, V, D).

---

## Architecture Pipeline

```
Classical Image (Grayscale / RGB)
    ↓  Preprocess & Downscale to N×N (4×4 to 64×64)
    ↓  [Optional] LSQB Watermark Injection (tiled ASCII string into LSB)
NEQR Encoding           →  |I⟩ = (1/N) Σ |f(y,x)⟩|yx⟩  (Uniform superposition)
    ↓  Quantum Arnold Cat Map (SWAP/CSWAP on position registers)
Confusion Layer         →  Spatial coordinates permuted in superposition
    ↓  Deterministic CNOT Diffusion (seeded key bits)
Diffusion Layer         →  Color qubit states scrambled
    ↓  Statevector Extraction / QASM Simulation
Encrypted Image         →  High-entropy classical array
    ↓  Combined Inverse Circuit (U_diff† · U_conf†)
Decrypted Image         →  Lossless reconstruction (SSIM = 1.0)
    ↓  LSQB Extraction
Verified Watermark      →  Signature verified: "IS-2301020531"
```

---

## Supported Grid Sizes & Resource Profile

| Grid Size | Position Qubits | Color Qubits | Total Qubits | Statevector Dim | Typical Time |
|:---------:|:---------------:|:------------:|:------------:|:---------------:|:------------:|
| **4×4**   | 4 ($2+2$)       | 8            | 12           | 4,096           | ~0.3s        |
| **8×8**   | 6 ($3+3$)       | 8            | 14           | 16,384          | ~0.4s        |
| **16×16** | 8 ($4+4$)       | 8            | 16           | 65,536          | ~1.0s        |
| **32×32** | 10 ($5+5$)      | 8            | 18           | 262,144         | ~7.0s        |
| **64×64** | 12 ($6+6$)      | 8            | 20           | 1,048,576       | ~45s         |

---

## Security Benchmark Metrics

| Metric | Target / Ideal | Description |
|:-------|:--------------:|:------------|
| **NPCR (%)** | $\ge 99.6\%$ | Number of Pixels Change Rate |
| **UACI (%)** | $\approx 33.4\%$ | Unified Average Changing Intensity |
| **Entropy (bits)** | $\ge 7.9$ | Information entropy (8-bit max = 8.0) |
| **SSIM (Encrypt)** | $\approx 0.0$ | Structural similarity of ciphertext to plaintext |
| **SSIM (Decrypt)** | **$1.0000$** | Lossless round-trip statevector recovery |
| **Correlation ($H/V/D$)** | $\approx 0.0$ | Horizontal, vertical, and diagonal pixel correlation |

---

## Installation & Usage

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/true-qcrypt.git
cd true-qcrypt

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Launch Streamlit Web UI
streamlit run app.py
```

---

## Academic Compliance

- ✅ **Monolithic Quantum Processing**: No classical slicing, tiling, or chunk loops.
- ✅ **Hardware Independence**: Compatible with Qiskit 2.x and AerSimulator.
- ✅ **Reversible Unitary Circuitry**: Decryption mathematically proven through adjoint gate synthesis.
- ✅ **Zero Extra Quantum Cost**: LSQB watermark uses the existing $q_0$ basis without increasing circuit depth.
