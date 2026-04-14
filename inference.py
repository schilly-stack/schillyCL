from pathlib import Path
import cv2
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--weight", type=str, default="weights/schillyCL.pth")
parser.add_argument("--input",  type=str, default="data/test")
parser.add_argument("--output", type=str, default="data/test_output")
args = parser.parse_args()

MODEL_PATH = args.weight
INPUT_DIR  = Path(args.input)
OUTPUT_DIR = Path(args.output)

if torch.backends.mps.is_available():
    device = 'mps'  
elif torch.cuda.is_available():
    device = 'cuda' 
else:
    device = 'cpu'

print(f"Running on: {device}")

def resize_to_height(img, target_height=2200):
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)

def run_full_page_inference():

    model_path = args.weight
    input_folder = Path(args.input)
    output_folder = Path(args.output)   
    
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=1)
    
    upsampler = RealESRGANer(
        scale=1,
        model_path=model_path,
        model=model,
        tile=0,            
        pre_pad=10,
        half=True if device != "cpu" else False,
        device=device
    )

    for img_name in os.listdir(input_folder):
        if img_name.endswith(('.jpg', '.png' ,)):
            img_path = os.path.join(input_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = resize_to_height(img, target_height=2200)
            try:
                output, _ = upsampler.enhance(img, outscale=1)
                cv2.imwrite(str(output_folder / img_name), output)
                print(f"Done: {img_name}")
        
            except Exception as e:
                print(f"Error processing {img_name}: {e}")



if __name__ == '__main__':
    
    run_full_page_inference()