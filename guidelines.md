## 🎯 Mission Objective
Integrate Quantum Least Significant Qubit (LSQB) Steganography into the existing monolithic NEQR architecture. The system must embed a hidden binary payload into the lowest quantum color register without altering the visual fidelity of the image or breaking the existing encryption/decryption pipeline. 

**CRITICAL CONSTRAINT:** Do not alter the existing Arnold Cat Map logic, CNOT diffusion layer, Statevector measurement backend, or the 64×64 maximum resolution limit. The current pipeline is highly stable and must remain untouched.

---

## 💧 Feature Addition: LSQB Steganography (Quantum Watermarking)

1. **The Watermark Payload:**
   * Create a binary stream from the author's identity string: `"IS-2301020531"`.
   * Convert this string to its 8-bit binary ASCII representation.
   * Repeat this binary sequence dynamically to match the exact number of pixels in the selected grid ($N \times N$).
2. **Classical Preprocessing (State Preparation):**
   * Before building the NEQR circuit, iterate through the flattened 1D array of the image's grayscale pixels.
   * Bitwise-overwrite the least significant bit (LSB) of each 8-bit pixel value with the corresponding bit from the watermark sequence. 
   * This modified array is then passed into the standard NEQR multi-controlled quantum gate builder, effectively turning the $q_0$ color qubit into a hidden data carrier.
3. **Lossless Preservation:**
   * Because the existing diffusion and confusion layers use purely unitary, reversible gates, the $q_0$ state will naturally be scrambled during encryption and perfectly restored during the statevector decryption phase.

---

## 🖥️ UI & Streamlit Deliverables (`app.py`)

1. **Sidebar Toggle:** Add a checkbox in the "Architecture Settings" sidebar labeled: `Embed Quantum Watermark (LSQB)`.
2. **Watermark Extraction UI:** 
   * In the main UI, beneath the "Decrypted" image panel, add an expanding section titled **"🔍 Extract Quantum Watermark"**.
   * When the user clicks a button to extract, isolate the least significant bit of every pixel in the *decrypted* image array.
   * Group the bits into 8-bit chunks, decode them back into ASCII characters, and display the extracted string in a green success banner (e.g., `Signature verified: IS-2301020531...`).
3. **Security Metrics Integrity:** Ensure the watermarking process does not crash the existing SSIM, Entropy, or NPCR calculations.

---

## ⚡ Execution Instructions for AI
1. Update `utils/` or the pre-processing block in `crypto/encryptor.py` to handle the string-to-binary LSB injection.
2. Update the post-processing block in `crypto/decryptor.py` or `app.py` to handle LSB extraction and binary-to-string decoding.
3. Add the UI toggles.
4. Run a sanity check to ensure the gate depth for a 64×64 image remains around 31,000 gates and does not cause a segmentation fault.