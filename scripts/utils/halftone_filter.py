
import cv2
import numpy as np
from pathlib import Path
import argparse
import math

def smooth_midtones(img, threshold_low=40, threshold_high=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mid_mask = (gray > threshold_low) & (gray < threshold_high)
    smoothed = img.copy()
    for _ in range(3):
        smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=50, sigmaSpace=50)
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    result = img.copy()
    result[mid_mask] = cv2.addWeighted(smoothed, 0.6, blurred, 0.4, 0)[mid_mask]
    return result

def apply_halftone(img, lpi=60, angle=45, dpi=300):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape

    # Cell size in pixels based on LPI and DPI
    cell = int(dpi / lpi)
    half = cell // 2

    # Work at 4x for anti-aliased dots
    scale = 4
    gh, gw = h * scale, w * scale
    cell_s = cell * scale
    half_s = cell_s // 2

    gray_up = cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_LINEAR)

    # Output starts as white
    output = np.ones((gh, gw), dtype=np.float32)

    # Edge mask — preserve ink lines
    gray_uint8 = (gray * 255).astype(np.uint8)
    edges = cv2.Canny(gray_uint8, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
    midtone_mask = (gray_uint8 > 25) & (gray_uint8 < 225) & (edges_dilated == 0)
    mask_up = cv2.resize(midtone_mask.astype(np.uint8), (gw, gh),
                         interpolation=cv2.INTER_NEAREST)

    # Preserve original blacks
    dark_mask = gray_uint8 < 25
    dark_up = cv2.resize(dark_mask.astype(np.uint8), (gw, gh),
                         interpolation=cv2.INTER_NEAREST)
    output[dark_up == 1] = 0.0

    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    diag = int(np.sqrt(gh**2 + gw**2)) + cell_s * 2
    centers = []

    for i in range(-diag // cell_s, diag // cell_s + 1):
        for j in range(-diag // cell_s, diag // cell_s + 1):
            gx = i * cell_s
            gy = j * cell_s
            rx = int(gx * cos_a - gy * sin_a) + gw // 2
            ry = int(gx * sin_a + gy * cos_a) + gh // 2
            if -half_s <= rx < gw + half_s and -half_s <= ry < gh + half_s:
                centers.append((rx, ry))

    for cx, cy in centers:
        x1 = max(0, cx - half_s)
        x2 = min(gw, cx + half_s)
        y1 = max(0, cy - half_s)
        y2 = min(gh, cy + half_s)

        if x2 <= x1 or y2 <= y1:
            continue

        cell_mask = mask_up[y1:y2, x1:x2]
        if np.mean(cell_mask) < 0.5:
            continue

        patch = gray_up[y1:y2, x1:x2]
        avg = float(np.mean(patch))

        if avg > 0.92:
            continue

        # Correct AM screening — dot area proportional to darkness
        import math
        cell_area = math.pi * (half_s ** 2)
        target_area = cell_area * (1.0 - avg)
        radius = int(math.sqrt(target_area / math.pi))

        if radius < 1:
            continue

        # Draw anti-aliased dot
        cv2.circle(output, (cx, cy), radius, 0.0, -1, lineType=cv2.LINE_AA)

    # Downscale — anti-aliasing happens naturally here
    output = cv2.resize(output, (w, h), interpolation=cv2.INTER_AREA)
    output = (output * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

def process(img_path, output_path, lpi, angle):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not load: {img_path.name}")
        return
    img = smooth_midtones(img)
    result = apply_halftone(img, lpi=lpi, angle=angle)
    success = cv2.imwrite(str(output_path), result)
    print(f"{'Done' if success else 'FAILED to write'}: {img_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    type=str, default="data/test_output")
    parser.add_argument("--output",   type=str, default="data/halftone_output")
    parser.add_argument("--lpi",   type=int, default=60)
    parser.add_argument("--angle", type=int, default=45)
    args = parser.parse_args()

    INPUT_DIR  = Path(args.input)
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.png"))
    )
    print(f"Found {len(files)} files\n")

    for img_path in files:
        process(img_path, OUTPUT_DIR / img_path.name, args.lpi, args.angle)

    print(f"\nFinished — output saved to {OUTPUT_DIR}")