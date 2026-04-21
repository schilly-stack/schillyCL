import zipfile
import os
from pathlib import Path
import argparse

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

def unzip_to_chapter_folder(zip_path, base_output):
    zip_path = Path(zip_path)
    # This creates the Chapter Folder (e.g., .../data/scans/SourceA_1145)
    chapter_folder = Path(base_output) / zip_path.stem
    chapter_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing Chapter: {zip_path.stem}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            filename = os.path.basename(member)
            
            # Skip internal folders and non-images
            if not filename or filename.startswith('.') or \
               Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            
            # We don't need the prefix on the filename anymore 
            # because the folder name handles it!
            target_path = chapter_folder / filename
            
            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                target.write(source.read())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    # Use the case-insensitive fix just in case
    zip_files = [f for f in in_path.iterdir() if f.suffix.lower() == '.zip']
    
    if not zip_files:
        print(f"No ZIPs found in {in_path}")
    else:
        for z in zip_files:
            unzip_to_chapter_folder(z, args.output)
        print("\nFinished! Each chapter has its own clean folder.")