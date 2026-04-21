import os
import random
import shutil
from pathlib import Path

base_dir = Path(r"C:\Users\Johan Bachmann\Documents\schillyCL\data\crops")
train_clean = base_dir / "clean"
train_deg = base_dir / "degraded"
val_clean = base_dir / "val" / "clean"
val_deg = base_dir / "val" / "degraded"

PERCENTAGE = 0.06 

val_clean.mkdir(parents=True, exist_ok=True)
val_deg.mkdir(parents=True, exist_ok=True)

files = [f for f in os.listdir(train_clean) if f.endswith(('.jpg', '.png', '.webp'))]
num_to_move = int(len(files) * PERCENTAGE)

print(f"📦 Total crops: {len(files)}")
print(f"🚚 Moving {num_to_move} pairs to validation...")

to_move = random.sample(files, num_to_move)

for filename in to_move:
    src_c = train_clean / filename
    src_d = train_deg / filename
    
    if src_c.exists() and src_d.exists():
        shutil.move(str(src_c), str(val_clean / filename))
        shutil.move(str(src_d), str(val_deg / filename))
    else:
        print(f"Missing pair for {filename}, skipping...")

print("Done! Your training and validation sets are now perfectly synced.")