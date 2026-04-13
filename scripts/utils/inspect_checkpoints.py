import torch
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python inspect_checkpoint.py <path_to_checkpoint.pth>")
    sys.exit(1)

checkpoint = torch.load(sys.argv[1], map_location="cpu")

print(type(checkpoint))
if isinstance(checkpoint, dict):
    print("Keys:", checkpoint.keys())
    for k, v in checkpoint.items():
        if hasattr(v, 'shape'):
            print(f"{k}: {v.shape}")
        elif isinstance(v, dict):
            for k2, v2 in list(v.items())[:3]:
                print(f"  {k2}: {v2.shape}")