"""
Aerial Image Processing Pipeline
---------------------------------
Phases:
  A - RGB channel splitting
  B - Red channel histogram
  C - Salt-and-pepper noise injection (manual)
  D - Manual pixel audit (5x5 window, hand-calculated median)
  E - Manual 3x3 median filter (no cv2/scipy/skimage)
  F - Texture sensitivity analysis
  G - Object detection (Otsu + BFS connected components)

Dependencies: numpy, Pillow, matplotlib, tkinter (built-in)
"""

import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_name(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def create_output_dirs(base="outputs"):
    dirs = {}
    for phase in ["phaseA", "phaseB", "phaseC", "phaseD", "phaseE", "phaseF", "phaseG"]:
        path = os.path.join(base, phase)
        os.makedirs(path, exist_ok=True)
        dirs[phase] = path
    print(f"[INFO] Outputs → {os.path.abspath(base)}")
    return dirs


# ---------------------------------------------------------------------------
# Phase A — RGB Channel Splitting
# ---------------------------------------------------------------------------

def split_channels(filepath, out_dir):
    img = Image.open(filepath).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    name = base_name(filepath).lower()
    channels = {}
    for idx, colour in enumerate(["red", "green", "blue"]):
        ch = arr[:, :, idx]
        channels[colour] = ch
        Image.fromarray(ch).save(os.path.join(out_dir, f"{name}_{colour}.png"))
        print(f"  [A] {colour} channel saved")
    return channels


# ---------------------------------------------------------------------------
# Phase B — Red Channel Histogram
# ---------------------------------------------------------------------------

def generate_histogram(channels, image_name, out_dir):
    red = channels["red"].flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(red, bins=256, range=(0, 255), color="crimson", alpha=0.8, edgecolor="none")
    ax.set_title(f"Red Channel Histogram — {image_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("DN Value (0–255)")
    ax.set_ylabel("Pixel Count")
    ax.set_xlim([0, 255])
    ax.axvline(red.mean(), color="navy", linestyle="--", linewidth=1.5, label=f"Mean={red.mean():.1f}")
    ax.axvline(float(np.median(red)), color="darkorange", linestyle=":", linewidth=1.5,
               label=f"Median={int(np.median(red))}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{image_name.lower()}_red_histogram.png"), dpi=150)
    plt.close(fig)
    print(f"  [B] Histogram saved  (mean={red.mean():.1f}, median={int(np.median(red))})")


# ---------------------------------------------------------------------------
# Phase C — Salt-and-Pepper Noise
# ---------------------------------------------------------------------------

def add_salt_pepper_noise(filepath, noise_ratio, out_dir):
    img = Image.open(filepath).convert("RGB")
    arr = np.array(img, dtype=np.uint8).copy()
    rng = np.random.default_rng(seed=42)
    rand_map = rng.random(arr.shape[:2])
    pepper_mask = rand_map < (noise_ratio / 2.0)
    salt_mask   = (rand_map >= noise_ratio / 2.0) & (rand_map < noise_ratio)
    arr[pepper_mask] = 0
    arr[salt_mask]   = 255
    name = base_name(filepath).lower()
    out_path = os.path.join(out_dir, f"noisy_{name}.png")
    Image.fromarray(arr).save(out_path)
    total = arr.shape[0] * arr.shape[1]
    print(f"  [C] Noisy image saved  (salt={salt_mask.sum():,}, pepper={pepper_mask.sum():,}, "
          f"total={salt_mask.sum()+pepper_mask.sum():,}/{total:,})")
    return arr


# ---------------------------------------------------------------------------
# Phase D — Manual Pixel Audit
# ---------------------------------------------------------------------------

def manual_pixel_audit(noisy_array, image_name, row, col, out_dir):
    H, W = noisy_array.shape[:2]
    grey = (0.299 * noisy_array[:, :, 0] +
            0.587 * noisy_array[:, :, 1] +
            0.114 * noisy_array[:, :, 2]).astype(np.uint8)

    r0, r1 = max(row - 2, 0), min(row + 3, H)
    c0, c1 = max(col - 2, 0), min(col + 3, W)
    window = grey[r0:r1, c0:c1]
    flat   = window.flatten().tolist()
    sorted_vals = sorted(flat)
    n = len(sorted_vals)
    median_val = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2.0

    lines = [
        "=" * 60,
        f"  MANUAL PIXEL AUDIT — {image_name.upper()}",
        "=" * 60,
        f"  Centre pixel : row={row}, col={col}",
        f"  Window size  : {window.shape[0]}x{window.shape[1]}",
        "",
        "  5x5 DN VALUE MATRIX:",
        "  +" + "-" * 27 + "+",
    ]
    for r in range(window.shape[0]):
        lines.append("  | " + "  ".join(f"{int(window[r, c]):3d}" for c in range(window.shape[1])) + " |")
    lines += [
        "  +" + "-" * 27 + "+",
        "",
        "  SORTED VALUES: " + str(sorted_vals),
        "",
        f"  MEDIAN (hand-calculated): {int(median_val)}",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print(report)

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    im = axes[0].imshow(window, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"5x5 Window @ ({row},{col})", fontweight="bold")
    for rr in range(window.shape[0]):
        for cc in range(window.shape[1]):
            v = int(window[rr, cc])
            axes[0].text(cc, rr, str(v), ha="center", va="center", fontsize=9,
                         color="white" if v < 128 else "black", fontweight="bold")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(im, ax=axes[0])

    axes[1].bar(range(n), sorted_vals, color="steelblue", edgecolor="navy")
    axes[1].axhline(median_val, color="red", linestyle="--", linewidth=2,
                    label=f"Median={int(median_val)}")
    axes[1].set_title("Sorted DN Values", fontweight="bold")
    axes[1].set_xlabel("Rank"); axes[1].set_ylabel("DN Value")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.4)
    plt.suptitle(f"Phase D — {image_name.upper()}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{image_name.lower()}_audit_chart.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, f"{image_name.lower()}_audit.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"  [D] Audit saved  (median={int(median_val)})")


# ---------------------------------------------------------------------------
# Phase E — Manual Median Filter
# ---------------------------------------------------------------------------

def manual_median_filter(noisy_array, image_name, out_dir, kernel_size=3):
    H, W, C = noisy_array.shape
    pad = kernel_size // 2
    padded   = np.pad(noisy_array, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    filtered = np.zeros_like(noisy_array)
    t0 = time.time()

    for ch in range(C):
        ch_pad = padded[:, :, ch]
        ch_out = filtered[:, :, ch]
        for i in range(H):
            for j in range(W):
                flat = ch_pad[i:i+kernel_size, j:j+kernel_size].flatten().tolist()
                flat.sort()
                ch_out[i, j] = flat[len(flat) // 2]
        print(f"  [E] Channel {ch+1}/{C} done ({time.time()-t0:.1f}s)")

    out_path = os.path.join(out_dir, f"filtered_{image_name.lower()}.png")
    Image.fromarray(filtered).save(out_path)

    # MSE & PSNR
    orig = np.array(Image.open(
        os.path.join(os.path.dirname(out_dir), "..", f"{image_name.lower()}_source.png")
        if False else out_path  # placeholder — computed in main
    ))
    print(f"  [E] Filtered image saved")
    return filtered


def compute_metrics(original_arr, filtered_arr):
    o = original_arr.astype(np.float64)
    f = filtered_arr.astype(np.float64)
    mse  = np.mean((o - f) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")
    return mse, psnr


# ---------------------------------------------------------------------------
# Phase F — Texture Sensitivity Analysis
# ---------------------------------------------------------------------------

_EDGE_THRESH   = 15.0
_DETAIL_THRESH = 40.0


def _mean_gradient(arr):
    grey = arr.mean(axis=2).astype(np.float32)
    return float((np.abs(np.diff(grey, axis=1)).mean() + np.abs(np.diff(grey, axis=0)).mean()) / 2.0)


def texture_sensitivity_analysis(results, out_dir):
    lines = ["=" * 70, "  TEXTURE SENSITIVITY ANALYSIS", "=" * 70, ""]
    metric_rows = []

    for name, data in results.items():
        og = _mean_gradient(data["original"])
        fg = _mean_gradient(data["filtered"])
        os_ = float(data["original"].std())
        fs_ = float(data["filtered"].std())
        gl  = og - fg

        lines += [
            f"  {name.upper()}",
            "  " + "-" * 50,
            f"  Gradient  orig={og:.2f}  filtered={fg:.2f}  loss={gl:.2f}",
            f"  Std-dev   orig={os_:.2f}  filtered={fs_:.2f}",
            "",
        ]

        if og >= _EDGE_THRESH:
            lines.append(f"  • Strong structural edges detected (gradient={og:.1f}).")
        else:
            lines.append(f"  • Relatively smooth texture (gradient={og:.1f}).")

        if gl < 2.0:
            lines.append(f"  • Edges well preserved after filtering (loss={gl:.2f}).")
        elif gl < 6.0:
            lines.append(f"  • Moderate edge smoothing occurred (loss={gl:.2f}).")
        else:
            lines.append(f"  • Significant detail loss after filtering (loss={gl:.2f}).")

        lines.append("")
        metric_rows.append((name, og, fg, os_, fs_))

    # Cross-class
    sorted_m = sorted(metric_rows, key=lambda x: x[1], reverse=True)
    lines += [
        "  CROSS-CLASS COMPARISON",
        "  " + "-" * 50,
        f"  Highest edge content : {sorted_m[0][0].upper()}",
        f"  Lowest edge content  : {sorted_m[-1][0].upper()}",
        "=" * 70,
    ]

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, "texture_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # Bar chart
    names_ = [r[0] for r in metric_rows]
    x = np.arange(len(names_))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    b1 = axes[0].bar(x - 0.2, [r[1] for r in metric_rows], 0.38, label="Original", color="steelblue")
    b2 = axes[0].bar(x + 0.2, [r[2] for r in metric_rows], 0.38, label="Filtered",  color="tomato")
    axes[0].set_title("Mean Gradient Magnitude", fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels([n.upper() for n in names_])
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.5)
    for bar in list(b1) + list(b2):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f"{bar.get_height():.1f}", ha="center", fontsize=9)

    b3 = axes[1].bar(x - 0.2, [r[3] for r in metric_rows], 0.38, label="Original", color="mediumseagreen")
    b4 = axes[1].bar(x + 0.2, [r[4] for r in metric_rows], 0.38, label="Filtered",  color="coral")
    axes[1].set_title("Pixel Std-Dev (Texture Richness)", fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels([n.upper() for n in names_])
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.5)
    for bar in list(b3) + list(b4):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f"{bar.get_height():.1f}", ha="center", fontsize=9)

    plt.suptitle("Phase F — Texture Analysis: Before vs After Filtering", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "texture_metrics.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  [F] Texture analysis saved")


# ---------------------------------------------------------------------------
# Phase G — Object Detection
# ---------------------------------------------------------------------------

def detect_objects(filtered_array, image_name, out_dir, min_area=500):
    # 1. Greyscale
    grey = (0.299 * filtered_array[:, :, 0].astype(np.float32) +
            0.587 * filtered_array[:, :, 1].astype(np.float32) +
            0.114 * filtered_array[:, :, 2].astype(np.float32)).astype(np.uint8)

    # 2. Manual Otsu thresholding
    hist      = np.bincount(grey.flatten(), minlength=256)
    total_px  = grey.size
    sum_total = np.dot(np.arange(256), hist)
    best_t, best_var, sum_bg, w_bg = 0, 0.0, 0.0, 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0: continue
        w_fg = total_px - w_bg
        if w_fg == 0: break
        sum_bg += t * hist[t]
        bv = w_bg * w_fg * (sum_bg/w_bg - (sum_total-sum_bg)/w_fg) ** 2
        if bv > best_var:
            best_var = bv; best_t = t
    binary = (grey > best_t).astype(np.uint8)

    # 3. Morphological opening (erode → dilate)
    def _erode(img):
        p = np.pad(img, 1, constant_values=0)
        o = np.ones_like(img)
        for di in range(3):
            for dj in range(3):
                o = np.minimum(o, p[di:di+img.shape[0], dj:dj+img.shape[1]])
        return o

    def _dilate(img):
        p = np.pad(img, 1, constant_values=0)
        o = np.zeros_like(img)
        for di in range(3):
            for dj in range(3):
                o = np.maximum(o, p[di:di+img.shape[0], dj:dj+img.shape[1]])
        return o

    opened = _dilate(_erode(binary))

    # 4. BFS connected-component labelling
    H, W      = opened.shape
    label_map = np.zeros((H, W), dtype=np.int32)
    cur_lbl   = 0
    for sr in range(H):
        for sc in range(W):
            if opened[sr, sc] == 1 and label_map[sr, sc] == 0:
                cur_lbl += 1
                q = [(sr, sc)]
                label_map[sr, sc] = cur_lbl
                while q:
                    r, c = q.pop()
                    for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                        if 0<=nr<H and 0<=nc<W and opened[nr,nc]==1 and label_map[nr,nc]==0:
                            label_map[nr,nc] = cur_lbl; q.append((nr,nc))

    # 5. Bounding boxes (top 5 by area)
    bboxes = []
    for lbl in range(1, cur_lbl+1):
        pos  = np.argwhere(label_map == lbl)
        area = len(pos)
        if area < min_area: continue
        r0,c0 = pos.min(axis=0); r1,c1 = pos.max(axis=0)
        bboxes.append((int(r0),int(c0),int(r1),int(c1),area))
    bboxes.sort(key=lambda x: x[4], reverse=True)
    bboxes = bboxes[:5]

    # 6. Draw bounding boxes
    result_img = Image.fromarray(filtered_array.copy())
    draw       = ImageDraw.Draw(result_img)
    for idx,(r0,c0,r1,c1,area) in enumerate(bboxes):
        col = ["red","lime","cyan","yellow","magenta"][idx % 5]
        draw.rectangle([(c0,r0),(c1,r1)], outline=col, width=3)
        draw.text((c0+4,r0+4), f"#{idx+1} A={area}", fill=col)

    result_img.save(os.path.join(out_dir, f"detected_{image_name.lower()}.png"))
    print(f"  [G] Otsu={best_t}, {len(bboxes)} objects detected")
    return binary, opened


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------

def save_comparison_figure(original_path, noisy_arr, filtered_arr, detected_path, image_name, out_dir):
    orig = np.array(Image.open(original_path).convert("RGB"))
    det  = np.array(Image.open(detected_path).convert("RGB"))
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (img, title) in zip(axes, [
        (orig,         "Original"),
        (noisy_arr,    "Noisy"),
        (filtered_arr, "Filtered (Manual Median)"),
        (det,          "Object Detection"),
    ]):
        ax.imshow(img); ax.set_title(title, fontweight="bold"); ax.axis("off")
    plt.suptitle(image_name.upper(), fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{image_name.lower()}_pipeline.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def select_images():
    root = tk.Tk(); root.withdraw()
    n_str = simpledialog.askstring("Image Processing Pipeline",
                                   "How many images to process?", parent=root)
    if not n_str:
        messagebox.showerror("Cancelled", "No input. Exiting."); sys.exit(0)
    try:
        n = int(n_str.strip()); assert n >= 1
    except:
        messagebox.showerror("Error", "Enter a positive integer."); sys.exit(1)

    paths = []
    for i in range(1, n+1):
        p = filedialog.askopenfilename(title=f"Select image {i}/{n}",
                                       filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All", "*.*")])
        if not p:
            messagebox.showerror("Cancelled", f"No file for image {i}."); sys.exit(0)
        paths.append(p)
    root.destroy()
    return paths


def ask_noise_ratio():
    while True:
        try:
            v = float(input("Noise ratio (0–1, e.g. 0.05): ").strip())
            if 0 < v < 1: return v
        except ValueError: pass
        print("  Invalid. Try again.")


def ask_pixel_coords(name, H, W):
    print(f"\n  [D] Centre pixel for {name}  (valid: row 2–{H-3}, col 2–{W-3})")
    while True:
        try:
            r = int(input("  Row: ").strip()); c = int(input("  Col: ").strip())
            if 0 <= r < H and 0 <= c < W: return r, c
        except ValueError: pass
        print("  Out of range.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    image_paths   = select_images()
    noise_ratio   = ask_noise_ratio()
    dirs          = create_output_dirs("outputs")
    results_for_F = {}

    for filepath in image_paths:
        name = base_name(filepath).lower()
        print(f"\n{'='*60}\n  {name}\n{'='*60}")

        original_arr = np.array(Image.open(filepath).convert("RGB"), dtype=np.uint8)
        H, W = original_arr.shape[:2]

        # A
        channels = split_channels(filepath, dirs["phaseA"])

        # B
        generate_histogram(channels, name, dirs["phaseB"])

        # C
        noisy_arr = add_salt_pepper_noise(filepath, noise_ratio, dirs["phaseC"])

        # D
        row, col = ask_pixel_coords(name, H, W)
        manual_pixel_audit(noisy_arr, name, row, col, dirs["phaseD"])

        # E
        print("  [E] Filtering — may take a few minutes...")
        filtered_arr = manual_median_filter(noisy_arr, name, dirs["phaseE"])

        # MSE / PSNR
        mse_noisy,   psnr_noisy   = compute_metrics(original_arr, noisy_arr)
        mse_filtered, psnr_filtered = compute_metrics(original_arr, filtered_arr)
        print(f"  [E] MSE  noisy={mse_noisy:.2f}  filtered={mse_filtered:.2f}")
        print(f"  [E] PSNR noisy={psnr_noisy:.2f}dB  filtered={psnr_filtered:.2f}dB "
              f"(+{psnr_filtered-psnr_noisy:.2f}dB)")

        # G
        binary, opened = detect_objects(filtered_arr, name, dirs["phaseG"])

        # Comparison figure
        detected_path = os.path.join(dirs["phaseG"], f"detected_{name}.png")
        save_comparison_figure(filepath, noisy_arr, filtered_arr, detected_path, name, dirs["phaseG"])

        results_for_F[name] = {"original": original_arr, "noisy": noisy_arr, "filtered": filtered_arr}

    # F (after all images)
    texture_sensitivity_analysis(results_for_F, dirs["phaseF"])

    print(f"\n{'='*60}\n  Done. Outputs → {os.path.abspath('outputs')}\n{'='*60}")


if __name__ == "__main__":
    main()
