import os
import shutil
from pathlib import Path
import argparse

def align_data(scans_dir, digital_master_dir, output_dir):
    scans_dir = Path(scans_dir)
    digital_master_dir = Path(digital_master_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all chapter folders in scans (e.g., SourceA_1150)
    scan_folders = [f for f in scans_dir.iterdir() if f.is_dir()]
    
    print(f"Found {len(scan_folders)} scan chapters to match...")

    for s_folder in scan_folders:
        # 1. Identify the chapter ID (everything after 'SourceA_')
        # This handles SourceA_1150 -> 1150
        try:
            chapter_id = s_folder.name.split('_', 1)[1]
        except IndexError:
            print(f"!! Skipping {s_folder.name}: Name must follow 'Source_ID' format.")
            continue

        # 2. Find the digital master folder that contains this ID
        # It looks for a folder named exactly '1150' or containing '1150'
        digital_match = None
        for d_folder in digital_master_dir.iterdir():
            if d_folder.is_dir() and chapter_id == d_folder.name:
                digital_match = d_folder
                break
        
        if digital_match:
            # 3. Create the new aligned folder (e.g., data/aligned_digitals/SourceA_1150)
            target_folder = output_dir / s_folder.name
            
            if not target_folder.exists():
                print(f"✅ Matching {s_folder.name} <---> Digital {chapter_id}")
                shutil.copytree(digital_match, target_folder)
            else:
                print(f"-- {target_folder.name} already exists. Skipping.")
        else:
            print(f"❌ Could not find digital master for chapter: {chapter_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans", type=str, required=True, help="Folder with SourceA_... folders")
    parser.add_argument("--masters", type=str, required=True, help="Folder with original digital chapters")
    parser.add_argument("--output", type=str, required=True, help="Where to save aligned digitals")
    args = parser.parse_args()

    align_data(args.scans, args.masters, args.output)
    print("\nAlignment complete! Your digital and scan folders now match 1:1.")