import cv2
import os
from pathlib import Path

def split_wide_pages(base_path):
    base_path = Path(base_path)
    
    # Supported image formats
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    print(f"Checking for spreads in: {base_path}")
    
    # Walk through all subfolders (SourceA_1180, etc.)
    for img_path in sorted(base_path.rglob("*")):
        if img_path.suffix.lower() not in extensions:
            continue
            
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # If width > height, it's a spread
        if w > h:
            print(f"✂️ Splitting spread: {img_path.name} ({w}x{h})")
            
            mid = w // 2
            
            # For most manga, the right side is the first page (Page A)
            # and the left side is the second page (Page B)
            # Adjust the order if your specific source is Western-style
            right_page = img[:, mid:]
            left_page  = img[:, :mid]
            
            # Create new filenames
            stem = img_path.stem
            ext = img_path.suffix
            
            # We save them with temp names so the Rename script can 
            # put them in the correct sequence later
            cv2.imwrite(str(img_path.parent / f"{stem}_part1{ext}"), right_page)
            cv2.imwrite(str(img_path.parent / f"{stem}_part2{ext}"), left_page)
            
            # Delete the original merged spread
            os.remove(img_path)

if __name__ == "__main__":
    scan_folder = r"C:\Users\Johan Bachmann\Documents\schillyCL\data\scans"
    split_wide_pages(scan_folder)
    print("\nDone! Spreads have been split. Remember to run your Mode 1 Rename script now!")