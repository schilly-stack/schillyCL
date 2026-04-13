import cv2
from pathlib import Path

SPREAD_WIDTH = 2806
SPREAD_HEIGHT = 2048

def split_spread(img_path, output_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    
    h, w = img.shape[:2]
    
    if w != SPREAD_WIDTH or h != SPREAD_HEIGHT:
        return False
    
    mid = w // 2
    left  = img[:, :mid]
    right = img[:, mid:]
    
    stem = img_path.stem
    cv2.imwrite(str(output_dir / f"{stem}_L.jpg"), left)
    cv2.imwrite(str(output_dir / f"{stem}_R.jpg"), right)
    
    # Slet originalen
    img_path.unlink()
    print(f"  Split: {img_path.name}")
    return True

for chapter_dir in sorted(Path("data/digitals").iterdir()):
    if not chapter_dir.is_dir():
        continue
    print(f"\nKapitel: {chapter_dir.name}")
    for img_path in sorted(chapter_dir.glob("*.jpg")):
        split_spread(img_path, chapter_dir)

print("\nFærdig")