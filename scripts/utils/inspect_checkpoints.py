import torch

checkpoint = torch.load(
    "C:/Users/Johan Bachmann/Real-ESRGAN/experiments/manga_restore/models/net_g_30000.pth",
    map_location="cpu"
)

print(type(checkpoint))
if isinstance(checkpoint, dict):
    print("Keys:", checkpoint.keys())
    for k, v in checkpoint.items():
        if hasattr(v, 'shape'):
            print(f"{k}: {v.shape}")
        elif isinstance(v, dict):
            # Print første par lag
            for k2, v2 in list(v.items())[:3]:
                print(f"  {k2}: {v2.shape}")