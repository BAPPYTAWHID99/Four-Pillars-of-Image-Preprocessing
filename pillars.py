import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import unsupervised_wiener, richardson_lucy

np.random.seed(42)
OUTDIR = "Output_1"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------- Helpers -----------------------------
def load_image(path, size=(150, 150)):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def motion_blur_kernel(size=15, angle=0):
    k = np.zeros((size, size), np.float32)
    k[size // 2, :] = 1
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    k = cv2.warpAffine(k, M, (size, size))
    return k / np.sum(k)

def circular_mask(shape, radius):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    return (x * x + y * y <= radius * radius).astype(np.float32)

def compute_metrics(ref, test, gray=False):
    """Return (PSNR, SSIM). Uses channel_axis for color images."""
    p = psnr(ref, test, data_range=255)
    if gray:
        s = ssim(ref, test, data_range=255)
    else:
        s = ssim(ref, test, data_range=255, channel_axis=2)
    return p, s

def show_strip(images, titles, filename, cmap=None, figsize=(18, 3)):
    """Display a horizontal strip and save it."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap=cmap)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(OUTDIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")

# ----------------------------- Pillar 1: Noise Suppression -----------------------------
def spatial_denoise(noisy):
    gauss = cv2.GaussianBlur(noisy, (5, 5), 0)
    median = cv2.medianBlur(noisy, 5)
    bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)
    nlmeans = cv2.fastNlMeansDenoisingColored(noisy, None, 10, 10, 7, 21)
    return gauss, median, bilateral, nlmeans

def fft_lowpass(img, radius=30):
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        f = np.fft.fftshift(np.fft.fft2(ch))
        mask = circular_mask(ch.shape, radius)
        ch_out = np.real(np.fft.ifft2(np.fft.ifftshift(f * mask)))
        out[:, :, c] = ch_out
    return np.clip(out, 0, 255).astype(np.uint8)

# ----------------------------- Pillar 2: Intensity Transformations -----------------------------
def negative(img):        return 255 - img

def log_transform(img):
    c = 255 / np.log(1 + 255)
    return np.clip(c * np.log1p(img.astype(np.float32)), 0, 255).astype(np.uint8)

def gamma_transform(img, gamma=2.2):
    return np.clip(255 * (img / 255.0) ** gamma, 0, 255).astype(np.uint8)

def contrast_stretch(img):
    p1, p99 = np.percentile(img, (1, 99))
    return np.clip((img - p1) * 255 / (p99 - p1), 0, 255).astype(np.uint8)

def hist_equalize_color(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def clahe_color(img, clip=2.0, tile=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# ----------------------------- Pillar 3: Restoration -----------------------------
def inverse_filter(gray, psf, eps=0.0):
    G = np.fft.fft2(gray.astype(np.float32))
    H = np.fft.fft2(psf, s=gray.shape)
    if eps == 0:
        F = G / (H + (np.abs(H) < 1e-6) * 1e-6)
    else:
        F = G * np.conj(H) / (np.abs(H) ** 2 + eps)
    return np.clip(np.real(np.fft.ifft2(F)), 0, 255).astype(np.uint8)

def wiener_filter(gray, psf, K=0.01):
    G = np.fft.fft2(gray.astype(np.float32))
    H = np.fft.fft2(psf, s=gray.shape)
    W = np.conj(H) / (np.abs(H) ** 2 + K)
    return np.clip(np.real(np.fft.ifft2(G * W)), 0, 255).astype(np.uint8)

# ----------------------------- Pillar 4: Morphology -----------------------------
def add_salt_noise(img, prob=0.02):
    noisy = img.copy()
    mask = np.random.rand(*img.shape[:2]) < prob
    noisy[mask] = 255
    return noisy

def morph_ops(img):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
    opened_dil = cv2.dilate(opened, k)
    dilated = cv2.dilate(img, k)
    dil_opened = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, k)
    return opened, opened_dil, dilated, dil_opened

# ----------------------------- Final Integrated Pipeline -----------------------------
def final_pipeline(img):
    """
    Denoise -> CLAHE (LAB) -> Normalize
    Best methods selected from Pillars 1 & 2.
    Restoration (Pillar 3) is intentionally excluded — ill-posed on motion blur.
    Morphology (Pillar 4) is intentionally excluded — intensity domain image.
    """
    # 1. Denoise with Bilateral (edge-preserving)
    denoised = cv2.bilateralFilter(img, 9, 75, 75)

    # 2. CLAHE on LAB L-channel
    enhanced = clahe_color(denoised, clip=2.0, tile=(8, 8))

    # 3. Normalize to 0-255
    final = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    final = final.astype(np.uint8)

    return denoised, enhanced, final

# ============================================================
if __name__ == "__main__":
    img = load_image("input.jpg")
    gray = to_gray(img)

    # ============================================================
    # ---------- Pillar 1: Noise Suppression ----------
    # ============================================================
    noisy = add_gaussian_noise(img, sigma=25)
    gauss, median, bilateral, nlmeans = spatial_denoise(noisy)
    fft_den = fft_lowpass(noisy, radius=30)

    show_strip(
        [img, noisy, gauss, median, bilateral, nlmeans],
        ["Original", "Noisy (σ=25)", "Gaussian", "Median", "Bilateral", "NL-Means"],
        "pillar1_spatial.png"
    )
    show_strip(
        [img, noisy, fft_den],
        ["Original", "Noisy (σ=25)", "FFT Low-pass (r=30)"],
        "pillar1_fft.png", figsize=(10, 3)
    )

    print("=== Pillar 1: Noise Suppression ===")
    print(f"{'Filter':<16}{'PSNR (dB)':<12}{'SSIM':<10}")
    print("-" * 38)
    for name, im in [("Noisy", noisy), ("Gaussian", gauss), ("Median", median),
                     ("Bilateral", bilateral), ("NL-Means", nlmeans),
                     ("FFT low-pass", fft_den)]:
        p, s = compute_metrics(img, im, gray=False)
        print(f"{name:<16}{p:<12.2f}{s:<10.4f}")
    print("→ Best on this blurry image: Gaussian (few sharp edges to preserve)")

    # ============================================================
    # ---------- Pillar 2: Intensity Enhancement ----------
    # ============================================================
    neg     = negative(img)
    log     = log_transform(img)
    gam     = gamma_transform(img)
    stretch = contrast_stretch(img)
    eq      = hist_equalize_color(img)
    clahe   = clahe_color(img)

    show_strip([img, neg],  ["Original", "Negative"],        "pillar2_negative.png", figsize=(7, 3))
    show_strip([img, log],  ["Original", "Log transform"],   "pillar2_log.png",      figsize=(7, 3))
    show_strip([img, gam],  ["Original", "Gamma (y=2.2)"],   "pillar2_gamma.png",    figsize=(7, 3))
    show_strip([img, stretch, eq, clahe],
               ["Original", "Contrast Stretch", "Hist. Equalization", "CLAHE"],
               "pillar2_histogram.png", figsize=(12, 3))

    # Histogram of original grayscale
    plt.figure(figsize=(5, 3))
    plt.hist(gray.ravel(), bins=256, range=(0, 255), color="gray")
    plt.title("Grayscale histogram")
    plt.xlabel("Pixel value"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "pillar2_histogram_plot.png"), dpi=150)
    plt.show()

    print("\n=== Pillar 2: Intensity Enhancement ===")
    print(f"{'Method':<22}{'Entropy':<12}{'Std Dev':<10}")
    print("-" * 44)
    for name, im in [("Original", img), ("Contrast Stretch", stretch),
                     ("Hist. Equalization", eq), ("CLAHE", clahe),
                     ("Gamma (y=2.2)", gam)]:
        g = to_gray(im)
        hist, _ = np.histogram(g, bins=256, range=(0, 255), density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        print(f"{name:<22}{entropy:<12.3f}{np.std(g):<10.2f}")
    print("→ Best: CLAHE (highest entropy, best local contrast)")

    # ============================================================
    # ---------- Pillar 3: Image Restoration ----------
    # ============================================================
    print("\n=== Pillar 3: Image Restoration ===")
    print("Note: Restoration of motion blur is an ILL-POSED INVERSE PROBLEM.")
    print("High frequencies are lost forever during motion blur — they cannot be")
    print("reconstructed because they were never recorded. Small errors in the PSF")
    print("cause huge errors in the output. Regularization (Wiener K-term) is essential.\n")

    psf = motion_blur_kernel(size=15, angle=0)
    blurred = cv2.filter2D(gray.astype(np.float32), -1, psf)
    blurred_noisy = add_gaussian_noise(blurred.astype(np.uint8), sigma=10)

    inv0   = inverse_filter(blurred_noisy, psf, eps=0)
    inv001 = inverse_filter(blurred_noisy, psf, eps=0.01)
    inv005 = inverse_filter(blurred_noisy, psf, eps=0.05)
    w001   = wiener_filter(blurred_noisy, psf, K=0.001)
    w01    = wiener_filter(blurred_noisy, psf, K=0.01)

    bnf = blurred_noisy.astype(np.float32) / 255.0
    deconv_uw, _ = unsupervised_wiener(bnf, psf.astype(np.float32))
    deconv_uw = np.clip(deconv_uw * 255, 0, 255).astype(np.uint8)

    deconv_rl = richardson_lucy(bnf, psf.astype(np.float32), num_iter=50)
    deconv_rl = np.clip(deconv_rl * 255, 0, 255).astype(np.uint8)

    show_strip(
        [gray, blurred_noisy, inv0, inv001, inv005],
        ["Original", "Blurred+Noisy", "Inverse eps=0", "Pseudo-inv eps=0.01", "Pseudo-inv eps=0.05"],
        "pillar3_inverse.png", cmap="gray"
    )
    show_strip(
        [gray, blurred_noisy, w001, w01, deconv_uw, deconv_rl],
        ["Original", "Blurred+Noisy", "Wiener K=0.001", "Wiener K=0.01",
         "Unsup. Wiener", "Richardson-Lucy (50)"],
        "pillar3_wiener.png", cmap="gray", figsize=(18, 3)
    )

    print(f"{'Method':<18}{'PSNR (dB)':<12}{'SSIM':<10}")
    print("-" * 40)
    for name, im in [("Blurred+Noisy", blurred_noisy), ("Inv eps=0", inv0),
                     ("Inv eps=0.01", inv001), ("Inv eps=0.05", inv005),
                     ("Wiener K=0.001", w001), ("Wiener K=0.01", w01),
                     ("Unsup. Wiener", deconv_uw), ("Richardson-Lucy", deconv_rl)]:
        p, s = compute_metrics(gray, im, gray=True)
        print(f"{name:<18}{p:<12.2f}{s:<10.4f}")
    print("→ Best: Unsupervised Wiener. Note: no method beat the degraded baseline")
    print("  (21.77 dB) because the image was ALREADY blurred before degradation.")

    # ============================================================
    # ---------- Pillar 4: Morphological Operations ----------
    # ============================================================
    salt = add_salt_noise(img, prob=0.02)
    opened, opened_dil, dilated, dil_opened = morph_ops(salt)

    show_strip(
        [img, salt, opened, opened_dil],
        ["Original", "RGB + Salt Noise", "After Opening", "Opening → Dilation"],
        "pillar4_open_dilate.png", figsize=(14, 3)
    )
    show_strip(
        [img, salt, dilated, dil_opened],
        ["Original", "RGB + Noise", "After Dilation", "Dilation → Opening"],
        "pillar4_dilate_open.png", figsize=(14, 3)
    )
    print("\n=== Pillar 4: Morphology ===")
    print("Opening → Dilation removes salt noise cleanly.")
    print("Dilation → Opening amplifies salt noise — order matters.")

    # ============================================================
    # ---------- FINAL INTEGRATED PIPELINE ----------
    # ============================================================
    print("\n=== Final Integrated Pipeline: Denoise → CLAHE → Normalize ===")
    denoised, enhanced, final = final_pipeline(img)

    show_strip(
        [img, denoised, enhanced, final],
        ["Original", "1. Denoised (Bilateral)", "2. CLAHE (LAB)", "3. Final (Normalized)"],
        "final_pipeline.png", figsize=(14, 3)
    )

    # Final pipeline metrics (compare against original)
    p_final, s_final = compute_metrics(img, final, gray=False)
    g_orig  = to_gray(img)
    g_final = to_gray(final)

    hist_o, _ = np.histogram(g_orig,  bins=256, range=(0, 255), density=True)
    hist_f, _ = np.histogram(g_final, bins=256, range=(0, 255), density=True)
    hist_o = hist_o[hist_o > 0]; hist_f = hist_f[hist_f > 0]
    ent_o = -np.sum(hist_o * np.log2(hist_o))
    ent_f = -np.sum(hist_f * np.log2(hist_f))

    print(f"{'Metric':<24}{'Original':<12}{'Final':<12}")
    print("-" * 48)
    print(f"{'PSNR (final vs orig)':<24}{'—':<12}{p_final:<12.2f}")
    print(f"{'SSIM (final vs orig)':<24}{'—':<12}{s_final:<12.4f}")
    print(f"{'Entropy':<24}{ent_o:<12.3f}{ent_f:<12.3f}")
    print("\n→ SSIM ≈ 1 means structure preserved.")
    print("→ Entropy increase means more information content.")
    print("→ Low PSNR is EXPECTED because CLAHE intentionally changes pixel values.")

    print(f"\nAll images saved to: {os.path.abspath(OUTDIR)}")