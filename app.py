"""
TrueQCrypt — Streamlit Web Interface (v3)
=========================================
Monolithic NEQR quantum image encryption with:
  - Grayscale & RGB color modes
  - 4x4 / 8x8 / 16x16 / 32x32 / 64x64 NEQR circuit sizes
  - Noiseless statevector reconstruction -> guaranteed SSIM = 1.0 decryption
  - Per-channel progress bars
  - Security metrics table with ideal-value annotations
  - pylatexenc-powered quantum circuit diagrams

Usage
-----
    cd true-qcrypt
    streamlit run app.py

Segfault prevention
-------------------
Qiskit-Aer's native C++ library (OpenMP) conflicts with Streamlit's
file-watcher threads, causing a segmentation fault at startup.
Mitigations applied here (env vars must be set BEFORE any import of
qiskit_aer or numpy/OpenBLAS):
  - OMP_NUM_THREADS=1       -> single OpenMP thread, no fork-unsafe state
  - OPENBLAS_NUM_THREADS=1  -> same for OpenBLAS (used by numpy)
  - MKL_NUM_THREADS=1       -> same for Intel MKL
  - QISKIT_PARALLEL=FALSE   -> disables Qiskit's own process-pool
  - TOKENIZERS_PARALLELISM=false -> prevents HuggingFace tokeniser fork warning
"""

# Safety env-vars — MUST appear before any other import
import os
os.environ.setdefault("OMP_NUM_THREADS",          "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",     "1")
os.environ.setdefault("MKL_NUM_THREADS",          "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS",   "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",      "1")
os.environ.setdefault("QISKIT_PARALLEL",          "FALSE")
os.environ.setdefault("TOKENIZERS_PARALLELISM",   "false")

# Force non-interactive matplotlib backend before pyplot is imported
import matplotlib
matplotlib.use("Agg")

# Safe initialization for Qiskit's mimalloc allocator
try:
    from qiskit import QuantumCircuit, QuantumRegister
    _qr = QuantumRegister(1, "_safe_init")
    del _qr
except Exception:
    from qiskit import QuantumCircuit

import io
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from crypto.encryptor import (
    QuantumEncryptor,
    extract_lsqb_watermark,
    WATERMARK_ID,
)
from crypto.decryptor import QuantumDecryptor
from analysis.metrics import SecurityMetrics

# ──────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrueQCrypt - Monolithic NEQR Quantum Image Encryption",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Professional Scientific UI Theme Styling
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global Typography & Refinements */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Header Container */
.qc-header {
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    padding-bottom: 1.25rem;
    margin-bottom: 1.5rem;
}
.qc-title {
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.2;
    margin: 0;
}
.qc-subtitle {
    font-size: 0.95rem;
    color: #64748b;
    margin: 0.4rem 0 0.85rem 0;
    font-weight: 400;
    line-height: 1.5;
}
.qc-badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.5rem;
}
.qc-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.22rem 0.6rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(148, 163, 184, 0.1);
    color: #475569;
    border: 1px solid rgba(148, 163, 184, 0.22);
}

@media (prefers-color-scheme: dark) {
    .qc-subtitle { color: #94a3b8; }
    .qc-badge {
        background: rgba(148, 163, 184, 0.08);
        color: #cbd5e1;
        border-color: rgba(148, 163, 184, 0.2);
    }
}

/* KPI Summary Cards */
[data-testid="stMetric"] {
    background: rgba(148, 163, 184, 0.04);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-weight: 600 !important;
    color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* Dataframe clean borders */
[data-testid="stDataFrame"] {
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.15);
}

/* Sidebar clean refinements */
[data-testid="stSidebar"] h1 {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
[data-testid="stSidebar"] h3 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
    color: #64748b;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

/* Button & interactive styling */
button[kind="primary"] {
    border-radius: 5px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("TrueQCrypt")
    st.caption("Monolithic NEQR · Arnold Cat Map · Deterministic CNOT Diffusion")
    st.divider()

    st.subheader("Architecture Settings")

    color_mode = st.selectbox(
        "Color Mode",
        options=["Grayscale", "RGB"],
        index=0,
        help="RGB runs the full NEQR pipeline on R, G, and B channels independently.",
    )

    block_size = st.selectbox(
        "Quantum Block Size (NEQR Circuit)",
        options=[4, 8, 16, 32, 64],
        index=2,
        help=(
            "Side length of the monolithic NEQR circuit.\n"
            "4x4 = 12 qubits | 8x8 = 14 qubits | 16x16 = 16 qubits | "
            "32x32 = 18 qubits | 64x64 = 20 qubits"
        ),
    )

    st.subheader("Simulation Settings")

    sim_mode = st.radio(
        "Reconstruction Backend",
        options=["Statevector (Exact, Noiseless)", "QASM Shot-Sampling (Legacy)"],
        index=0,
        help=(
            "Statevector — uses Aer statevector simulator to extract exact "
            "quantum amplitudes. No shot noise. SSIM = 1.0 on decryption. "
            "Works for up to 64x64.\n\n"
            "QASM — probabilistic shot sampling. Prone to missing pixels "
            "for larger resolutions."
        ),
    )
    use_statevector = sim_mode.startswith("Statevector")

    shots = 8192  # Only used in QASM mode
    if not use_statevector:
        shots = st.select_slider(
            "QASM Simulator Shots",
            options=[4096, 8192, 16384, 32768, 65536],
            value=16384,
            help="Higher shots reduce pixel dropout for large images.",
        )

    cat_iterations = st.slider(
        "Arnold Cat Map Iterations",
        min_value=1, max_value=5, value=3,
        help="More iterations yield deeper spatial scrambling."
    )

    key_seed = st.number_input(
        "Diffusion Key Seed",
        min_value=0, max_value=999999, value=42, step=1,
        help="Pre-shared pseudorandom seed. Must match between Encrypt and Decrypt.",
    )

    st.divider()
    st.subheader("Authentication & Watermarking")

    embed_watermark = st.checkbox(
        "Embed LSQB Quantum Watermark",
        value=False,
        help=(
            "Overwrites the Least Significant Bit of pixels with the "
            "identity string 'IS-2301020531' (ASCII, tiled).\n\n"
            "The watermark is encoded directly into the q0 color qubit of the "
            "NEQR circuit. Because all encryption gates are unitary and reversible, "
            "the watermark is restored after statevector decryption "
            "(SSIM = 1.0, zero extra gates)."
        ),
    )

    st.divider()
    run_decrypt  = st.checkbox("Execute Decryption Pipeline (SSIM Verification)", value=True)
    show_circuit = st.checkbox("Render Quantum Circuit Diagram", value=False)

    st.divider()
    st.caption(
        "Academic Compliance: Monolithic NEQR, non-fragmented quantum registers.\n\n"
        "Unitary transformations executed inside Qiskit QuantumCircuit representations."
    )

# ──────────────────────────────────────────────────────────────────────
# Main Application View
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="qc-header">
    <h1 class="qc-title">TrueQCrypt</h1>
    <div class="qc-subtitle">
        Monolithic NEQR Quantum Image Encryption System with LSQB Steganographic Authentication
    </div>
    <div class="qc-badge-container">
        <span class="qc-badge">Monolithic NEQR</span>
        <span class="qc-badge">Arnold Cat Map</span>
        <span class="qc-badge">Deterministic CNOT Diffusion</span>
        <span class="qc-badge">Statevector Simulation</span>
        <span class="qc-badge">SSIM = 1.0 Lossless</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("Pipeline Architecture & Operational Specification", expanded=False):
    st.markdown(
        """
        The **TrueQCrypt** framework implements a non-chunked, monolithic quantum encryption pipeline:
        
        1. **LSQB Steganography Injection** — Classical preprocessing embeds authentication payload into least significant pixel bits.
        2. **Monolithic NEQR State Preparation** — Maps spatial coordinates to $2n$ position qubits and intensity to 8 color qubits via normalized superposition.
        3. **Quantum Arnold Cat Map (Confusion Layer)** — Applies SWAP and CSWAP gates on position registers to iteratively scramble spatial coordinates.
        4. **Deterministic CNOT Diffusion (Diffusion Layer)** — Flips color qubits using a cryptographically seeded pseudo-random bitstream.
        5. **Statevector Amplitude Extraction** — Directly extracts exact probability amplitudes without shot noise, ensuring unitary reversibility ($\text{SSIM} = 1.0$).
        """
    )

uploaded = st.file_uploader(
    "Select Source Image",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    help="Upload an image file to process through the quantum encryption pipeline.",
)

if not uploaded:
    st.info("Upload an image file to begin the quantum encryption pipeline.")
    st.stop()

pil_image = Image.open(uploaded)
color_mode_str = color_mode.lower()

st.divider()
col_btn, col_meta = st.columns([1, 3])
with col_btn:
    run_btn = st.button("Execute Quantum Encryption", type="primary", use_container_width=True)

with col_meta:
    preview_size = pil_image.size
    st.markdown(
        f"**Source File:** `{uploaded.name}` · **Native Resolution:** `{preview_size[0]}×{preview_size[1]}` "
        f"· **Color Space:** `{pil_image.mode}`\n\n"
        f"**Target Quantum Resolution:** `{block_size}×{block_size}` · **Operating Mode:** `{color_mode}`"
    )

# Initialize session state for caching results across button clicks
if "enc_result" not in st.session_state:
    st.session_state["enc_result"] = None
if "wm_extracted" not in st.session_state:
    st.session_state["wm_extracted"] = False

# Reset cache if a new image is uploaded
file_sig = f"{uploaded.name}_{uploaded.size}"
if st.session_state.get("current_file_sig") != file_sig:
    st.session_state["current_file_sig"] = file_sig
    st.session_state["enc_result"] = None
    st.session_state["wm_extracted"] = False

if run_btn:
    st.session_state["wm_extracted"] = False

    # ──────────────────────────────────────────────────────────────────
    # Encryption
    # ──────────────────────────────────────────────────────────────────
    channels_needed = ["R", "G", "B"] if color_mode_str == "rgb" else ["L"]
    total_channels = len(channels_needed)

    enc_progress = st.progress(0, text="Initializing...")
    enc_status = st.empty()

    def enc_progress_cb(ch_name: str, fraction: float):
        pct = int(fraction * 100)
        enc_progress.progress(pct, text=f"Encrypting channel {ch_name} ({pct}%)...")

    try:
        enc_status.info(
            "Constructing NEQR circuit and extracting statevector..."
            if use_statevector else
            "Constructing NEQR circuit and running QASM simulation..."
        )
        encryptor = QuantumEncryptor(
            pil_image,
            target_size=block_size,
            cat_iterations=cat_iterations,
            shots=shots,
            key_seed=int(key_seed),
            color_mode=color_mode_str,
            use_statevector=use_statevector,
            embed_watermark=embed_watermark,
        )
        encrypted = encryptor.encrypt(progress_cb=enc_progress_cb)
        enc_progress.progress(100, text="Encryption complete.")
        stats = encryptor.get_circuit_stats()
        original_img = encryptor.original_image
        enc_status.empty()
        enc_progress.empty()

    except Exception as exc:
        enc_progress.empty()
        enc_status.empty()
        st.error(f"Encryption failed: {exc}")
        st.exception(exc)
        st.stop()

    # ──────────────────────────────────────────────────────────────────
    # Optional decryption
    # ──────────────────────────────────────────────────────────────────
    decrypted = None
    if run_decrypt:
        dec_progress = st.progress(0, text="Decrypting...")
        dec_status = st.empty()

        def dec_progress_cb(ch_name: str, fraction: float):
            pct = int(fraction * 100)
            dec_progress.progress(pct, text=f"Decrypting channel {ch_name} ({pct}%)...")

        try:
            dec_status.info("Executing inverse quantum decryption circuit...")
            decryptor = QuantumDecryptor(
                original_image=original_img,
                encrypted_image=encrypted,
                cat_iterations=cat_iterations,
                shots=shots,
                key_seed=int(key_seed),
                color_mode=color_mode_str,
                use_statevector=use_statevector,
                mode="combined",
            )
            decrypted = decryptor.decrypt(progress_cb=dec_progress_cb)
            dec_progress.progress(100, text="Decryption complete.")
            dec_status.empty()
            dec_progress.empty()
        except Exception as exc:
            dec_progress.empty()
            dec_status.empty()
            st.warning(f"Decryption failed: {exc}")

    # ──────────────────────────────────────────────────────────────────
    # Security metrics
    # ──────────────────────────────────────────────────────────────────
    def _to_gray(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            return np.array(Image.fromarray(arr.astype(np.uint8)).convert("L"))
        return arr.astype(np.uint8)

    orig_g = _to_gray(original_img)
    enc_g  = _to_gray(encrypted)

    metrics_obj = SecurityMetrics(orig_g, enc_g)
    enc_metrics = metrics_obj.compute_all()
    enc_assessment = metrics_obj.get_assessment()

    dec_metrics = {}
    if decrypted is not None:
        dec_g = _to_gray(decrypted)
        dec_metrics_obj = SecurityMetrics(orig_g, dec_g)
        dec_metrics = dec_metrics_obj.compute_all()

    st.session_state["enc_result"] = {
        "original_img": original_img,
        "encrypted": encrypted,
        "decrypted": decrypted,
        "stats": stats,
        "enc_metrics": enc_metrics,
        "enc_assessment": enc_assessment,
        "dec_metrics": dec_metrics,
        "orig_g": orig_g,
        "enc_g": enc_g,
        "circuit": encryptor.get_circuit(),
        "block_size": block_size,
        "color_mode": color_mode,
        "use_statevector": use_statevector,
        "embed_watermark": embed_watermark,
    }

if st.session_state["enc_result"] is None:
    st.stop()

# Load saved results from session state
res = st.session_state["enc_result"]
original_img = res["original_img"]
encrypted = res["encrypted"]
decrypted = res["decrypted"]
stats = res["stats"]
enc_metrics = res["enc_metrics"]
enc_assessment = res["enc_assessment"]
dec_metrics = res["dec_metrics"]
orig_g = res["orig_g"]
enc_g = res["enc_g"]
qc_disp = res.get("circuit")
res_block_size = res.get("block_size", block_size)
res_color_mode = res.get("color_mode", color_mode)
res_use_sv = res.get("use_statevector", use_statevector)
res_embed_wm = res.get("embed_watermark", embed_watermark)

# ──────────────────────────────────────────────────────────────────────
# KPI Metrics Summary
# ──────────────────────────────────────────────────────────────────────
st.divider()

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
with kpi_c1:
    st.metric("Execution Time", f"{stats['elapsed_seconds']}s")
with kpi_c2:
    st.metric("Security Rating", enc_assessment)
with kpi_c3:
    st.metric("Quantum Register", f"{stats['num_qubits']} Qubits")
with kpi_c4:
    st.metric("Circuit Depth", str(stats["depth"]))

# ──────────────────────────────────────────────────────────────────────
# Image Panels
# ──────────────────────────────────────────────────────────────────────
DISP_SIZE = max(res_block_size * 12, 140)

def _to_pil_display(arr: np.ndarray) -> Image.Image:
    img = Image.fromarray(arr.astype(np.uint8))
    return img.resize((DISP_SIZE, DISP_SIZE), Image.NEAREST)

n_panels = 3 if decrypted is not None else 2
cols = st.columns(n_panels)

with cols[0]:
    st.subheader("Original Plaintext")
    st.image(_to_pil_display(original_img), use_container_width=True,
             caption=f"Input: {res_block_size}×{res_block_size} · Mode: {res_color_mode}")

with cols[1]:
    st.subheader("Encrypted Ciphertext")
    caption_enc = f"Quantum Statevector ({res_block_size}×{res_block_size})" if res_use_sv else f"QASM Shot Sampling ({res_block_size}×{res_block_size})"
    st.image(_to_pil_display(encrypted), use_container_width=True, caption=caption_enc)
    buf = io.BytesIO()
    Image.fromarray(encrypted.astype(np.uint8)).save(buf, format="PNG")
    st.download_button("Download Encrypted Image", buf.getvalue(),
                       "encrypted_qcrypt.png", "image/png", use_container_width=True)

if decrypted is not None:
    with cols[2]:
        st.subheader("Decrypted Plaintext")
        ssim_val = dec_metrics.get('SSIM', '—')
        caption_dec = f"SSIM Fidelity: {ssim_val} (Exact Unitary Recovery)"
        st.image(_to_pil_display(decrypted), use_container_width=True, caption=caption_dec)

# ──────────────────────────────────────────────────────────────────────
# LSQB Watermark Extraction & Verification
# ──────────────────────────────────────────────────────────────────────
if decrypted is not None and res_embed_wm:
    st.divider()
    with st.expander("LSQB Watermark Extraction & Verification", expanded=True):
        st.markdown(
            "Extracts the **Least Significant Bit (LSB)** from each pixel address in the "
            "decrypted image matrix and reconstructs the embedded ASCII ownership signature."
        )
        if st.button("Extract Watermark Payload"):
            st.session_state["wm_extracted"] = True

        if st.session_state.get("wm_extracted", False):
            dec_gray = (
                np.array(Image.fromarray(decrypted.astype(np.uint8), mode="RGB").convert("L"))
                if decrypted.ndim == 3 else decrypted.astype(np.uint8)
            )
            wm = extract_lsqb_watermark(dec_gray, expected=WATERMARK_ID)
            cap = wm["capacity_chars"]
            full_match = cap >= len(WATERMARK_ID)
            if wm["verified"]:
                label = "Full Match" if full_match else f"Partial · {cap}/{len(WATERMARK_ID)} Chars"
                st.success(f"Signature Verified ({label}): `{wm['signature']}`")
                st.caption(f"Decoded Byte Stream: `{wm['decoded_str'][:80]}`")
            else:
                st.error(
                    f"Signature Mismatch — Extracted: `{wm['signature']}` "
                    f"| Expected: `{WATERMARK_ID[:cap]}`"
                )
            col_l, col_r = st.columns(2)
            col_l.metric("Extracted LSB Bits", len(wm["raw_bits"]))
            col_r.metric("Decoded Characters", cap)

# ──────────────────────────────────────────────────────────────────────
# Cryptographic Security Metrics
# ──────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Cryptographic Security Metrics")

IDEAL_ENC = {
    "NPCR (%)":                ("≥ 99.60%", "Pixel sensitivity rate"),
    "UACI (%)":                ("≈ 33.40%", "Average intensity difference"),
    "Entropy (bits)":          ("≥ 7.90",   "Information randomness (max 8.0)"),
    "Original Entropy (bits)": ("Informational", "Source plaintext randomness"),
    "SSIM":                    ("≈ 0.000",  "Structural dissimilarity"),
    "Correlation (H)":         ("≈ 0.000",  "Horizontal pixel correlation"),
    "Correlation (V)":         ("≈ 0.000",  "Vertical pixel correlation"),
    "Correlation (D)":         ("≈ 0.000",  "Diagonal pixel correlation"),
}

IDEAL_DEC = {
    "NPCR (%)":                ("0.00%",    "Exact match"),
    "UACI (%)":                ("0.00%",    "Zero intensity deviation"),
    "Entropy (bits)":          ("Source",   "Identical entropy to source"),
    "Original Entropy (bits)": ("Informational", "Source baseline"),
    "SSIM":                    ("1.0000",   "Lossless structural identity"),
    "Correlation (H)":         ("Source",   "Matches source correlation"),
    "Correlation (V)":         ("Source",   "Matches source correlation"),
    "Correlation (D)":         ("Source",   "Matches source correlation"),
}

col_enc, col_dec = st.columns(2)
with col_enc:
    st.markdown("**Encryption Evaluation** *(Plaintext vs Ciphertext)*")
    rows_enc = []
    for k, v in enc_metrics.items():
        bench, desc = IDEAL_ENC.get(k, ("—", ""))
        val_str = f"{v:.4f}" if isinstance(v, (float, int)) else str(v)
        rows_enc.append({
            "Metric": k,
            "Measured Value": val_str,
            "Theoretical Target": bench,
            "Description": desc,
        })
    st.dataframe(pd.DataFrame(rows_enc), hide_index=True, use_container_width=True)

with col_dec:
    st.markdown("**Decryption Verification** *(Plaintext vs Decrypted)*")
    if dec_metrics:
        rows_dec = []
        for k, v in dec_metrics.items():
            bench, desc = IDEAL_DEC.get(k, ("—", ""))
            val_str = f"{v:.4f}" if isinstance(v, (float, int)) else str(v)
            rows_dec.append({
                "Metric": k,
                "Measured Value": val_str,
                "Theoretical Target": bench,
                "Description": desc,
            })
        st.dataframe(pd.DataFrame(rows_dec), hide_index=True, use_container_width=True)
    else:
        st.info("Enable 'Execute Decryption Pipeline' in the sidebar to compute decryption fidelity metrics.")

# ──────────────────────────────────────────────────────────────────────
# Circuit Execution Parameters
# ──────────────────────────────────────────────────────────────────────
with st.expander("Quantum Circuit Execution Parameters", expanded=False):
    c_s1, c_s2, c_s3, c_s4 = st.columns(4)
    c_s1.metric("Qubit Count", stats.get("num_qubits", "—"))
    c_s2.metric("Circuit Depth", stats.get("depth", "—"))
    c_s3.metric("Quantum Gates", stats.get("gate_count", "—"))
    c_s4.metric("Backend Type", "Statevector" if res_use_sv else "QASM Simulator")
    
    st.markdown("**Detailed Gate Breakdown & Allocation:**")
    st.json(stats)

# ──────────────────────────────────────────────────────────────────────
# Pixel Intensity Distributions (Histograms)
# ──────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Pixel Intensity Distributions")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), dpi=120)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#94a3b8')
    ax.spines['bottom'].set_color('#94a3b8')
    ax.tick_params(colors='#64748b', labelsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#94a3b8')
    ax.set_axisbelow(True)

axes[0].hist(orig_g.flatten(), bins=32, range=(0, 256), color="#2563eb", alpha=0.85, edgecolor="#1d4ed8", linewidth=0.5)
axes[0].set_title("Original Plaintext Distribution", fontsize=9.5, fontweight='600', pad=8, color='#334155')
axes[0].set_xlabel("Pixel Intensity Level [0-255]", fontsize=8, color='#64748b')
axes[0].set_ylabel("Frequency Count", fontsize=8, color='#64748b')

axes[1].hist(enc_g.flatten(), bins=32, range=(0, 256), color="#4f46e5", alpha=0.85, edgecolor="#4338ca", linewidth=0.5)
axes[1].set_title("Encrypted Ciphertext Distribution (Uniform)", fontsize=9.5, fontweight='600', pad=8, color='#334155')
axes[1].set_xlabel("Pixel Intensity Level [0-255]", fontsize=8, color='#64748b')
axes[1].set_ylabel("Frequency Count", fontsize=8, color='#64748b')

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────
# Quantum Circuit Diagram
# ──────────────────────────────────────────────────────────────────────
if show_circuit:
    st.divider()
    st.subheader("Quantum Circuit Diagram")
    
    try:
        qc_disp = res.get("circuit")
        if qc_disp is not None:
            total_ops = len(qc_disp.data)
            
            c_info1, c_info2, c_info3 = st.columns(3)
            c_info1.metric("Total Circuit Operations", f"{total_ops:,}")
            c_info2.metric("Quantum Registers", f"{qc_disp.num_qubits} Qubits")
            c_info3.metric("Circuit Depth", f"{qc_disp.depth():,}")
            
            if total_ops > 40:
                st.caption(
                    f"The full monolithic circuit contains {total_ops:,} quantum gates across {qc_disp.num_qubits} qubits. "
                    "Rendering thousands of gates simultaneously can exceed maximum image memory dimensions. "
                    "A representative schematic sequence is visualized below."
                )
                preview_ops = st.slider(
                    "Operations to display in schematic preview",
                    min_value=10,
                    max_value=min(100, total_ops),
                    value=min(30, total_ops),
                    step=10,
                    help="Adjust the number of sequential quantum gate operations displayed in the schematic."
                )
                
                # Build safe sub-circuit for schematic display
                sub_qc = QuantumCircuit(*qc_disp.qregs, *qc_disp.cregs)
                for instr in qc_disp.data[:preview_ops]:
                    sub_qc.append(instr.operation, instr.qubits, instr.clbits)
                
                fig_c = sub_qc.draw(output="mpl", fold=30, scale=0.8, style={"backgroundcolor": "#FFFFFF"})
            else:
                st.caption("Rendering full monolithic quantum circuit schematic.")
                fig_c = qc_disp.draw(output="mpl", fold=30, scale=0.8, style={"backgroundcolor": "#FFFFFF"})
                
            st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)
        else:
            st.info("No circuit available to display.")
    except Exception as e:
        st.warning(f"Circuit visualization notice: {e}")

st.divider()
st.caption(
    "TrueQCrypt Framework · Monolithic NEQR State Preparation · "
    "Quantum Permutation & Diffusion · Lossless Statevector Reconstruction"
)
