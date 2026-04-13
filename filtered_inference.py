import cv2
import torch
import numpy as np
from pathlib import Path
from basicsr.archs.rrdbnet_arch import RRDBNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "weights/schillyCL.pth"
INPUT_DIR  = Path("data/test")
OUTPUT_DIR = Path("data/test_output")

model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=1
)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

model.load_state_dict(checkpoint["params_ema"], strict=True)
model.eval().to(device)

def preprocess(img):
    return cv2.bilateralFilter(img, d=5, sigmaColor=20, sigmaSpace=20)

def smooth_halftone(img, blur_strength=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mid_mask = (gray > 60) & (gray < 200)
    
    blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
    
    result = img.copy()
    result[mid_mask] = blurred[mid_mask]
    
    return result

def adaptive_blacks(img, block_size=11, C=8):
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     thresh = cv2.adaptiveThreshold(
         gray,
         255,
         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
         cv2.THRESH_BINARY,
         block_size,
         C
     )     
     dark_mask = gray < 150
     result = img.copy()
     result[dark_mask & (thresh == 0)] = [0, 0, 0]
     return result

def post_process(img):
    img = img.astype(np.float32)
    img = (img - 30) / (230 - 30) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = smooth_halftone(img, blur_strength=3)
    img = adaptive_blacks(img, block_size=11, C=8)
    return img

def process_tile(tile):
    t = torch.from_numpy(tile).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (out * 255).clip(0, 255).astype(np.uint8)

def make_weight_mask(tile_size):
    center = tile_size / 2
    Y, X = np.ogrid[:tile_size, :tile_size]
    dist = np.sqrt((X - center)**2 + (Y - center)**2)
    mask = np.exp(-0.5 * (dist / (tile_size / 4))**2)
    mask = mask / mask.max()
    return mask[:, :, np.newaxis].astype(np.float32)

def restore_tiled(img, tile_size=256, overlap=64):
    h, w = img.shape[:2]
    output = np.zeros(img.shape, dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    weight_mask = make_weight_mask(tile_size)

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            x1 = x
            y1 = y
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)

            tile = img[y1:y2, x1:x2]
            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

            result = process_tile(tile)
            result = result[:y2-y1, :x2-x1]

            wm = weight_mask[:y2-y1, :x2-x1]
            output[y1:y2, x1:x2] += result * wm
            weight[y1:y2, x1:x2] += wm

    output = (output / weight).clip(0, 255).astype(np.uint8)
    return output

input_dir  = Path("data/test")
output_dir = Path("data/test_output")
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob("*.png"):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = preprocess(img)
    result = restore_tiled(img)
    result = post_process(result)
    cv2.imwrite(str(output_dir / img_path.name), result)
    print(f"Done: {img_path.name}")

print("Done")