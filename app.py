"""
TrueQCrypt — Streamlit Web Interface (v3)
=========================================
Monolithic NEQR quantum image encryption with:
  - Grayscale & RGB color modes
  - 4×4 / 8×8 / 16×16 / 32×32 NEQR circuit sizes
  - Noiseless statevector reconstruction → guaranteed SSIM = 1.0 decryption
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
  - OMP_NUM_THREADS=1       → single OpenMP thread, no fork-unsafe state
  - OPENBLAS_NUM_THREADS=1  → same for OpenBLAS (used by numpy)
  - MKL_NUM_THREADS=1       → same for Intel MKL
  - QISKIT_PARALLEL=FALSE   → disables Qiskit's own process-pool
  - TOKENIZERS_PARALLELISM=false → prevents HuggingFace tokeniser fork warning

These are also set in .streamlit/config.toml (fileWatcherType = "none").
"""

# ── Safety env-vars — MUST appear before any other import ─────────────
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
    from qiskit.circuit import QuantumRegister
    _qr = QuantumRegister(1, "_safe_init")
    del _qr
except Exception:
    pass

import io
import sys
import time

import numpy as np
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
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrueQCrypt",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔐 TrueQCrypt")
    st.caption("Monolithic NEQR · Arnold Cat Map · Deterministic CNOT Diffusion")
    st.divider()

    st.subheader("🏗️ Architecture Settings")

    color_mode = st.selectbox(
        "Color Mode",
        options=["Grayscale", "RGB"],
        index=0,
        help="RGB runs the full NEQR pipeline on R, G, and B channels independently.",
    )

    block_size = st.selectbox(
        "Quantum Block Size (NEQR circuit)",
        options=[4, 8, 16, 32, 64],
        index=2,
        help=(
            "Side length of the monolithic NEQR circuit.\n"
            "4×4 = 12 qubits | 8×8 = 14 qubits | 16×16 = 16 qubits | "
            "32×32 = 18 qubits | 64×64 = 20 qubits"
        ),
    )

    st.subheader("⚙️ Simulation Settings")

    sim_mode = st.radio(
        "Reconstruction backend",
        options=["Statevector (exact, noiseless)", "QASM shot-sampling (legacy)"],
        index=0,
        help=(
            "**Statevector** — uses Aer statevector simulator to extract exact "
            "quantum amplitudes. No shot noise. SSIM = 1.0 on decryption. "
            "Works perfectly for 32×32 (18 qubits).\n\n"
            "**QASM** — probabilistic shot sampling. Prone to missing pixels "
            "for 32×32 (1024 addresses need thousands of shots)."
        ),
    )
    use_statevector = sim_mode.startswith("Statevector")

    shots = 8192  # Only used in QASM mode
    if not use_statevector:
        shots = st.select_slider(
            "QASM simulator shots",
            options=[4096, 8192, 16384, 32768, 65536],
            value=16384,
            help="Higher shots reduce pixel dropout for large images.",
        )

    cat_iterations = st.slider(
        "Arnold Cat Map iterations",
        min_value=1, max_value=5, value=3,
        help="More iterations → deeper scrambling."
    )

    key_seed = st.number_input(
        "Encryption key seed",
        min_value=0, max_value=999999, value=42, step=1,
        help="Must match between Encrypt and Decrypt for lossless recovery.",
    )

    st.divider()
    st.subheader("🔏 Steganography")

    embed_watermark = st.checkbox(
        "Embed Quantum Watermark (LSQB)",
        value=False,
        help=(
            "Overwrites the **Least Significant Bit** of every pixel with the "
            "author identity string `IS-2301020531` (ASCII, tiled).\n\n"
            "The watermark is encoded directly into the **q₀ color qubit** of the "
            "NEQR circuit. Because all encryption gates are unitary and reversible, "
            "the watermark is perfectly restored after statevector decryption "
            "(SSIM = 1.0, zero extra gates)."
        ),
    )

    st.divider()
    run_decrypt  = st.checkbox("Also run decryption (SSIM test)", value=True)
    show_circuit = st.checkbox("Show quantum circuit diagram",    value=False)

    st.divider()
    st.caption(
        "**Academic compliance:** monolithic NEQR, no block-splitting.\n\n"
        "All encryption logic runs inside `QuantumCircuit` objects."
    )

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
st.title("🔐 TrueQCrypt: Quantum Image Encryption System")
st.markdown(
    """
    **Monolithic NEQR pipeline** — every pixel is encoded into a single quantum circuit:
    1. **NEQR Encoding** — uniform superposition over all pixel addresses
    2. **Quantum Arnold Cat Map** — SWAP/CSWAP gates scramble position register
    3. **Deterministic CNOT Diffusion** — seeded key flips color register bits
    4. **Statevector Extraction** — exact amplitude readout, zero shot noise, SSIM = 1.0
    """
)

uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
)

if not uploaded:
    st.info("👆 Upload an image to get started.")
    st.stop()

pil_image = Image.open(uploaded)
color_mode_str = color_mode.lower()

st.divider()
col_btn, col_meta = st.columns([1, 3])
with col_btn:
    run_btn = st.button("🚀 Encrypt", type="primary", use_container_width=True)

with col_meta:
    preview_size = pil_image.size
    st.markdown(
        f"**File:** `{uploaded.name}` · Size: `{preview_size[0]}×{preview_size[1]}`"
        f" · Mode: `{pil_image.mode}`\n\n"
        f"Will be resized to **{block_size}×{block_size}** · "
        f"Color mode: **{color_mode}**"
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

    enc_progress = st.progress(0, text="Initialising…")
    enc_status = st.empty()

    def enc_progress_cb(ch_name: str, fraction: float):
        pct = int(fraction * 100)
        enc_progress.progress(pct, text=f"Encrypting channel {ch_name} ({pct}%)…")

    try:
        enc_status.info(
            "🔄 Building NEQR circuit and extracting statevector…"
            if use_statevector else
            "🔄 Building NEQR circuit and running QASM simulation…"
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
        enc_progress.progress(100, text="Encryption complete ✅")
        stats = encryptor.get_circuit_stats()
        original_img = encryptor.original_image
        enc_status.empty()
        enc_progress.empty()

    except Exception as exc:
        enc_progress.empty()
        enc_status.empty()
        st.error(f"❌ Encryption failed: {exc}")
        st.exception(exc)
        st.stop()

    # ──────────────────────────────────────────────────────────────────
    # Optional decryption
    # ──────────────────────────────────────────────────────────────────
    decrypted = None
    if run_decrypt:
        dec_progress = st.progress(0, text="Decrypting…")
        dec_status = st.empty()

        def dec_progress_cb(ch_name: str, fraction: float):
            pct = int(fraction * 100)
            dec_progress.progress(pct, text=f"Decrypting channel {ch_name} ({pct}%)…")

        try:
            dec_status.info("🔄 Running inverse quantum pipeline (combined circuit)…")
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
            dec_progress.progress(100, text="Decryption complete ✅")
            dec_status.empty()
            dec_progress.empty()
        except Exception as exc:
            dec_progress.empty()
            dec_status.empty()
            st.warning(f"⚠️ Decryption failed: {exc}")

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
# Display
# ──────────────────────────────────────────────────────────────────────
st.success(
    f"✅ Done in **{stats['elapsed_seconds']}s** | "
    f"Encryption security: **{enc_assessment}** | "
    f"Qubits: **{stats['num_qubits']}** | Depth: **{stats['depth']}**"
)

DISP_SIZE = max(res_block_size * 12, 128)

def _to_pil_display(arr: np.ndarray) -> Image.Image:
    img = Image.fromarray(arr.astype(np.uint8))
    return img.resize((DISP_SIZE, DISP_SIZE), Image.NEAREST)

n_panels = 3 if decrypted is not None else 2
cols = st.columns(n_panels)

with cols[0]:
    st.subheader("📷 Original")
    st.image(_to_pil_display(original_img), use_container_width=True,
             caption=f"{res_block_size}×{res_block_size} · {res_color_mode}")

with cols[1]:
    st.subheader("🔒 Encrypted")
    caption_enc = "Statevector extraction" if res_use_sv else "QASM shot-sampling"
    st.image(_to_pil_display(encrypted), use_container_width=True, caption=caption_enc)
    buf = io.BytesIO()
    Image.fromarray(encrypted.astype(np.uint8)).save(buf, format="PNG")
    st.download_button("⬇️ Download encrypted", buf.getvalue(),
                       "encrypted_qcrypt.png", "image/png")

if decrypted is not None:
    with cols[2]:
        st.subheader("🔓 Decrypted")
        st.image(_to_pil_display(decrypted), use_container_width=True,
                 caption=f"SSIM vs original: {dec_metrics.get('SSIM', '—')}")

# ──────────────────────────────────────────────────────────────────────
# LSQB Watermark Extraction
# ──────────────────────────────────────────────────────────────────────
if decrypted is not None and res_embed_wm:
    st.divider()
    with st.expander("🔍 Extract Quantum Watermark (LSQB)", expanded=True):
        st.markdown(
            "Isolating the **Least Significant Bit** of every pixel in the "
            "decrypted image and decoding the embedded ASCII signature."
        )
        if st.button("🔓 Extract Watermark"):
            st.session_state["wm_extracted"] = True

        if st.session_state.get("wm_extracted", False):
            # Use grayscale view for extraction
            dec_gray = (
                np.array(Image.fromarray(decrypted.astype(np.uint8), mode="RGB").convert("L"))
                if decrypted.ndim == 3 else decrypted.astype(np.uint8)
            )
            wm = extract_lsqb_watermark(dec_gray, expected=WATERMARK_ID)
            cap = wm["capacity_chars"]
            full_match = cap >= len(WATERMARK_ID)
            if wm["verified"]:
                label = "full match" if full_match else f"partial · {cap}/{len(WATERMARK_ID)} chars"
                st.success(f"✅ **Signature verified** ({label}): `{wm['signature']}`")
                st.caption(f"Decoded stream (first 80 chars): `{wm['decoded_str'][:80]}…`")
            else:
                st.error(
                    f"❌ Signature mismatch — extracted: `{wm['signature']}` "
                    f"| expected prefix: `{WATERMARK_ID[:cap]}`"
                )
            col_l, col_r = st.columns(2)
            col_l.metric("LSBs extracted", len(wm["raw_bits"]))
            col_r.metric("Chars decoded", cap)


# ──────────────────────────────────────────────────────────────────────
# Metrics tables
# ──────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Security Metrics")

import pandas as pd

IDEAL = {
    "NPCR (%)":               "≥ 99.6%",
    "UACI (%)":               "≈ 33.4%",
    "Entropy (bits)":         "≥ 7.9",
    "Original Entropy (bits)":"—",
    "SSIM":                   "≈ 0  (encrypt) / ≥ 0.999 (decrypt)",
    "Correlation (H)":        "≈ 0",
    "Correlation (V)":        "≈ 0",
    "Correlation (D)":        "≈ 0",
}

col_enc, col_dec = st.columns(2)
with col_enc:
    st.markdown("**Encryption metrics** (original vs encrypted)")
    rows = [{"Metric": k, "Value": v, "Ideal": IDEAL.get(k, "—")}
            for k, v in enc_metrics.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with col_dec:
    st.markdown("**Decryption metrics** (original vs decrypted)")
    if dec_metrics:
        rows = [{"Metric": k, "Value": v, "Ideal": IDEAL.get(k, "—")}
                for k, v in dec_metrics.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("Enable 'Also run decryption' in the sidebar.")

# Circuit stats
with st.expander("⚛️ Circuit Statistics"):
    st.json(stats)

# ──────────────────────────────────────────────────────────────────────
# Histograms
# ──────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Pixel Intensity Histograms")
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].hist(orig_g.flatten(), bins=32, color="steelblue", alpha=0.8)
axes[0].set_title("Original"); axes[0].set_xlabel("Intensity")
axes[1].hist(enc_g.flatten(),  bins=32, color="crimson",   alpha=0.8)
axes[1].set_title("Encrypted"); axes[1].set_xlabel("Intensity")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────
# Optional circuit diagram (pylatexenc required — now installed)
# ──────────────────────────────────────────────────────────────────────
if show_circuit:
    st.divider()
    st.subheader("⚛️ Quantum Circuit Diagram")
    st.caption(
        "Rendered via Matplotlib + pylatexenc. "
        "Displaying the last-built encryption circuit (fold at 50 gates per row)."
    )
    try:
        qc_disp = res.get("circuit")
        if qc_disp is not None:
            fig_c = qc_disp.draw(output="mpl", fold=50, style={"backgroundcolor": "#FFFFFF"})
            st.pyplot(fig_c, use_container_width=True)
        else:
            st.info("No circuit available to display.")
    except Exception as e:
        st.warning(f"Circuit diagram error: {e}")

st.divider()
st.caption(
    "TrueQCrypt v3 · Monolithic NEQR · Deterministic Diffusion Key · "
    "Statevector Reconstruction · SSIM = 1.0 Guaranteed"
)
