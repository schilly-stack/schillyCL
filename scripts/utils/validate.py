# scripts/validate.py
import cv2
import numpy as np
from pathlib import Path

def side_by_side(clean_path, degraded_path, output_path):
    clean = cv2.imread(str(clean_path))
    degraded = cv2.imread(str(degraded_path))
    
    # Resize height to be the same for both images
    h = min(clean.shape[0], degraded.shape[0])
    clean = cv2.resize(clean, (int(clean.shape[1] * h / clean.shape[0]), h))
    degraded = cv2.resize(degraded, (int(degraded.shape[1] * h / degraded.shape[0]), h))
    
    comparison = np.hstack([clean, degraded])
    cv2.imwrite(str(output_path), comparison)

# Run validation
validate_dir = Path("data/validate")
validate_dir.mkdir(exist_ok=True)

clean_dir = Path("data/clean")
degraded_dir = Path("data/degraded")

for i, p in enumerate(list(clean_dir.glob("*.jpg"))[:5]):
    side_by_side(p, degraded_dir / p.name, validate_dir / f"compare_{i}.png")