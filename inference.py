import cv2
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def resize_to_height(img, target_height=2200):
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)

def run_full_page_inference():

    model_path = r"C:/Users/Johan Bachmann/Real-ESRGAN/experiments/manga_restore/models/net_g_latest.pth"
    input_folder = r'C:\Users\Johan Bachmann\Documents\manga-restore\data\test'
    output_folder = r'C:\Users\Johan Bachmann\Documents\manga-restore\data\test_output'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=1)
    
    upsampler = RealESRGANer(
        scale=1,
        model_path=model_path,
        model=model,
        tile=0,            
        pre_pad=10,
        half=True,
        device='cuda'
    )

    for img_name in os.listdir(input_folder):
        if img_name.endswith(('.jpg', '.png' ,)):
            img_path = os.path.join(input_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = resize_to_height(img, target_height=2200)
            output, _ = upsampler.enhance(img, outscale=1)
            cv2.imwrite(os.path.join(output_folder, img_name), output)
            print(f"Done: {img_name}")


if __name__ == '__main__':
    
    run_full_page_inference()