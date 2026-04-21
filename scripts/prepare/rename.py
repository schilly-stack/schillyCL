"""
Rename utilities for schillyCL dataset preparation.

Mode 1 — Sequential: Renames all jpg files in chapter subfolders
         to sequential numbers (001.jpg, 002.jpg, ...)
         Runs on both data/scans and data/digitals

Mode 2 — Flatten: Prefixes filenames with their parent folder name
         e.g. chapter_01/001.jpg → chapter_01_001.jpg
         Runs on data/digitals only
"""

from pathlib import Path

def rename_sequential(base_dirs):
    """
    Renames all jpg files in chapter subfolders to sequential numbers.
    Uses temp names first to avoid conflicts (e.g. from split spreads).
    """
    for base_dir in base_dirs:
        print(f"\nBehandler {base_dir.name}:")
        for chapter_dir in sorted(base_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
            
            files = sorted(chapter_dir.glob("*.jpg"))
            if not files:
                print(f"  {chapter_dir.name}: No files found")
                continue
            
            # Rename til temp navne først for at undgå konflikter
            temp_files = []
            for i, f in enumerate(files):
                temp = chapter_dir / f"__temp_{i:04d}.jpg"
                f.rename(temp)
                temp_files.append(temp)
            
            for i, f in enumerate(temp_files):
                new_name = chapter_dir / f"{i+1:03d}.jpg"
                f.rename(new_name)
            
            print(f"  {chapter_dir.name}: {len(files)} files renamed")

def rename_flatten(digital_dir):
    """
    Prefix file names with chapter folder name.
    e.g. 1147/001.jpg → 1147/1147_001.jpg
    Used to avoid name conflicts when files are moved to the same folder.
    """
    for img_path in digital_dir.rglob("*.jpg"):
        if img_path.name.startswith(img_path.parent.name + "_"):
            continue
        new_name = f"{img_path.parent.name}_{img_path.name}"
        new_path = img_path.parent / new_name
        img_path.rename(new_path)
        print(f"  {img_path.name} → {new_name}")

if __name__ == "__main__":
    print("schillyCL — Rename Utility")
    print("==========================")
    print("1 — Sequential rename (scans + digitals, for alignment pipeline)")
    print("2 — Flatten rename (digitals, for syntetisk degradering pipeline)")
    print()
    
    choice = input("Choose mode (1/2): ").strip()
    
    if choice == "1":
        rename_sequential([Path("data/scans"), Path("data/aligned_digitals")])
        print("\nDone — run align_and_crop.py next")
    
    elif choice == "2":
        rename_flatten(Path("data/aligned_digitals"))
        print("\nDone — run degrade.py next")
    
    else:
        print("Invalid choice. Ending...")