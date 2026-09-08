"""
Cryptographic Security Metrics
===============================
Standard evaluation metrics for assessing the security and quality of the
quantum-encrypted image relative to the original.

Metrics Implemented
-------------------
1. **NPCR (Number of Pixels Change Rate)**
   Measures the percentage of pixels that differ between the original and
   encrypted images.  Ideal value: ~99.6% (maximum pixel sensitivity).

       NPCR = (Σ D(i,j) / (M×N)) × 100%
       where D(i,j) = 0 if O(i,j) = E(i,j), else 1.

2. **UACI (Unified Average Changing Intensity)**
   Measures the average intensity difference between the original and
   encrypted images, normalised to [0,1].  Ideal value: ~33.4%.

       UACI = (1 / (M×N)) × Σ |O(i,j) - E(i,j)| / 255 × 100%

3. **Shannon Entropy**
   Measures the information randomness of the encrypted image.  For a
   perfectly random 8-bit image, the theoretical maximum is 8.0 bits.
   Values ≥ 7.9 indicate strong encryption.

       H(X) = -Σ p(x_i) × log2(p(x_i))   (summed over 256 intensity levels)

4. **SSIM (Structural Similarity Index)**
   Measures structural similarity between original and encrypted images.
   A value close to 0 (or negative) indicates the encryption has destroyed
   structural information, which is desirable.

5. **Correlation Coefficient (Adjacent Pixel)**
   Measures the statistical correlation between horizontally, vertically,
   and diagonally adjacent pixels.  Strong encryption yields values near 0.

       corr(x, y) = cov(x,y) / (σ_x × σ_y)
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from skimage.metrics import structural_similarity as ssim


class SecurityMetrics:
    """
    Computes cryptographic security metrics for encrypted image evaluation.

    Parameters
    ----------
    original : np.ndarray
        Original grayscale image, uint8, 2-D.
    encrypted : np.ndarray
        Encrypted grayscale image, uint8, 2-D (same shape as original).
    """

    def __init__(self, original: np.ndarray, encrypted: np.ndarray) -> None:
        if original.shape != encrypted.shape:
            raise ValueError(
                f"Shape mismatch: original {original.shape} vs "
                f"encrypted {encrypted.shape}."
            )
        self.original = original.astype(np.float64)
        self.encrypted = encrypted.astype(np.float64)
        self._M, self._N = original.shape

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def npcr(self) -> float:
        """
        Number of Pixels Change Rate (%).

        Returns
        -------
        float
            NPCR value in [0, 100]. Ideal: ~99.6%.
        """
        diff = (self.original != self.encrypted).astype(np.float64)
        return float(np.sum(diff) / (self._M * self._N) * 100.0)

    def uaci(self) -> float:
        """
        Unified Average Changing Intensity (%).

        Returns
        -------
        float
            UACI value in [0, 100]. Ideal: ~33.4%.
        """
        intensity_diff = np.abs(self.original - self.encrypted) / 255.0
        return float(np.mean(intensity_diff) * 100.0)

    def entropy(self, image: np.ndarray | None = None) -> float:
        """
        Shannon entropy of an image (bits per pixel).

        Parameters
        ----------
        image : np.ndarray, optional
            Image to compute entropy on. Defaults to `self.encrypted`.

        Returns
        -------
        float
            Entropy in bits. Maximum for 8-bit image: 8.0.
        """
        if image is None:
            image = self.encrypted
        img_uint8 = image.astype(np.uint8)
        # Compute histogram over 256 intensity levels
        hist, _ = np.histogram(img_uint8, bins=256, range=(0, 255))
        # Mask zero bins to avoid log(0)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob)))

    def ssim_score(self) -> float:
        """
        Structural Similarity Index between original and encrypted image.

        For images smaller than 7×7, win_size is set to the largest odd
        integer that fits within the image dimensions.

        Returns
        -------
        float
            SSIM in [-1, 1]. Near 0 (or negative) is ideal for encryption.
        """
        o = self.original.astype(np.uint8)
        e = self.encrypted.astype(np.uint8)
        min_side = min(o.shape)
        # win_size must be odd and ≤ min_side; default is 7
        win_size = min(7, min_side) if min(7, min_side) % 2 == 1 else min(7, min_side) - 1
        win_size = max(win_size, 3)  # must be at least 3
        score, _ = ssim(o, e, full=True, data_range=255, win_size=win_size)
        return float(score)

    def correlation(self, direction: str = "horizontal") -> float:
        """
        Adjacent-pixel correlation coefficient.

        Parameters
        ----------
        direction : str
            One of 'horizontal', 'vertical', 'diagonal'.

        Returns
        -------
        float
            Correlation in [-1, 1]. Near 0 is ideal for encryption.
        """
        img = self.encrypted
        if direction == "horizontal":
            x = img[:, :-1].flatten()
            y = img[:, 1:].flatten()
        elif direction == "vertical":
            x = img[:-1, :].flatten()
            y = img[1:, :].flatten()
        elif direction == "diagonal":
            x = img[:-1, :-1].flatten()
            y = img[1:, 1:].flatten()
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return float(np.corrcoef(x, y)[0, 1])

    # ------------------------------------------------------------------
    # Aggregated report
    # ------------------------------------------------------------------

    def compute_all(self) -> dict:
        """
        Compute all metrics and return as a structured dictionary.

        Returns
        -------
        dict
            Keys: metric name → value (float, rounded to 4 decimal places).
        """
        return {
            "NPCR (%)": round(self.npcr(), 4),
            "UACI (%)": round(self.uaci(), 4),
            "Entropy (bits)": round(self.entropy(), 4),
            "Original Entropy (bits)": round(self.entropy(self.original.astype(np.uint8)), 4),
            "SSIM": round(self.ssim_score(), 4),
            "Correlation (H)": round(self.correlation("horizontal"), 4),
            "Correlation (V)": round(self.correlation("vertical"), 4),
            "Correlation (D)": round(self.correlation("diagonal"), 4),
        }

    def get_assessment(self) -> str:
        """
        Return a human-readable security assessment based on metric thresholds.

        Returns
        -------
        str
            'Strong', 'Moderate', or 'Weak'.
        """
        results = self.compute_all()
        entropy_ok = results["Entropy (bits)"] >= 7.5
        npcr_ok = results["NPCR (%)"] >= 90.0
        ssim_ok = abs(results["SSIM"]) <= 0.3

        if entropy_ok and npcr_ok and ssim_ok:
            return "Strong"
        elif sum([entropy_ok, npcr_ok, ssim_ok]) >= 2:
            return "Moderate"
        else:
            return "Weak"
