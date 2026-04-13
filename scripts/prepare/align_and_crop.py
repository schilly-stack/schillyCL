# scripts/prepare/align_and_crop.py

import cv2
import numpy as np
from pathlib import Path
import random

PATCH_SIZE = 512
CROPS_PER_IMAGE = 10

def autocrop_scan(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_means = np.mean(gray, axis=1)
    content_rows = np.where(row_means < 240)[0]
    if len(content_rows) == 0:
        return img
    top    = content_rows[0]
    bottom = content_rows[-1]
    col_means = np.mean(gray, axis=0)
    content_cols = np.where(col_means < 240)[0]
    if len(content_cols) == 0:
        return img
    left  = content_cols[0]
    right = content_cols[-1]
    return img[top:bottom, left:right]

def match_resolution(scan, digital):
    h_d = digital.shape[0]
    h_s = scan.shape[0]
    scale = h_d / h_s
    new_w = int(scan.shape[1] * scale)
    return cv2.resize(scan, (new_w, h_d), interpolation=cv2.INTER_AREA)

def align_pair(scan, digital):
    s_gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    d_gray = cv2.cvtColor(digital, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(s_gray, None)
    kp2, des2 = orb.detectAndCompute(d_gray, None)
    if des1 is None or des2 is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:200]
    if len(matches) < 10:
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        return None
    det = np.linalg.det(M[:2, :2])
    if det < 0.5 or det > 2.0:
        return None
    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    if abs(angle) > 15:
        return None
    h, w = digital.shape[:2]
    aligned = cv2.warpPerspective(scan, M, (w, h))
    return aligned

def is_valid_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    if mean_val > 230 or mean_val < 20:
        return False
    if np.std(gray) < 20:
        return False
    return True

def random_crops(aligned_scan, digital, clean_out, degraded_out, stem):
    h, w = digital.shape[:2]
    if h < PATCH_SIZE or w < PATCH_SIZE:
        print(f"  Too small for crops: {stem}")
        return 0
    count = 0
    for i in range(CROPS_PER_IMAGE):
        x = random.randint(0, w - PATCH_SIZE)
        y = random.randint(0, h - PATCH_SIZE)
        clean_crop    = digital[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        degraded_crop = aligned_scan[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        if not is_valid_crop(degraded_crop):
            continue
        cv2.imwrite(str(clean_out    / f"{stem}_crop{i:02d}.jpg"), clean_crop)
        cv2.imwrite(str(degraded_out / f"{stem}_crop{i:02d}.jpg"), degraded_crop)
        count += 1
    return count

scan_dir     = Path("data/scans")
digital_dir  = Path("data/digitals")
clean_out    = Path("data/crops/clean")
degraded_out = Path("data/crops/degraded")
clean_out.mkdir(parents=True, exist_ok=True)
degraded_out.mkdir(parents=True, exist_ok=True)

total_crops = 0
skipped     = 0

for chapter_dir in sorted(scan_dir.iterdir()):
    if not chapter_dir.is_dir():
        continue
    digital_chapter = digital_dir / chapter_dir.name
    if not digital_chapter.exists():
        print(f"No matching digital folder: {chapter_dir.name}")
        continue
    print(f"\nProcessing chapter: {chapter_dir.name}")
    for scan_path in sorted(chapter_dir.glob("*.jpg")):
        digital_path = digital_chapter / scan_path.name
        if not digital_path.exists():
            matches = list(digital_chapter.rglob(scan_path.name))
            if not matches:
                print(f"  No match: {scan_path.name}")
                skipped += 1
                continue
            digital_path = matches[0]
        scan    = cv2.imread(str(scan_path))
        digital = cv2.imread(str(digital_path))
        if scan is None or digital is None:
            print(f"  Could not load: {scan_path.name}")
            skipped += 1
            continue
        scan    = autocrop_scan(scan)
        scan    = match_resolution(scan, digital)
        aligned = align_pair(scan, digital)
        if aligned is None:
            print(f"  Alignment failed: {scan_path.name}")
            skipped += 1
            continue
        stem = f"{chapter_dir.name}_{scan_path.stem}"
        n = random_crops(aligned, digital, clean_out, degraded_out, stem)
        total_crops += n
        print(f"  {scan_path.name} → {n} crops")

print(f"\nDone — {total_crops} crops generated, {skipped} pages skipped.")