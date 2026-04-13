import cv2
import numpy as np
from pathlib import Path
import random

# --- Indlæs patches ---
# Forvent 3 af hver type i patches/white/, patches/black/, patches/gray/
def load_patches(folder):
    patches = []
    for p in Path(folder).glob("*.png"):
        tex = cv2.imread(str(p)).astype(np.float32)
        baseline = np.mean(tex)
        patches.append(tex - baseline)
    return patches

white_patches = load_patches("patches/white")
black_patches = load_patches("patches/black")
gray_patches  = load_patches("patches/gray")

def tile_texture(texture, target_h, target_w):
    th, tw = texture.shape[:2]
    reps_y = (target_h // th) + 2
    reps_x = (target_w // tw) + 2
    tiled = np.tile(texture, (reps_y, reps_x, 1))
    oy = random.randint(0, th)
    ox = random.randint(0, tw)
    return tiled[oy:oy+target_h, ox:ox+target_w]

def stitch_texture(patches, target_h, target_w):
    """
    Vælg en tilfældig patch og tile den — 
    med tilfældig blending af en anden patch ovenpå for variation
    """
    primary = tile_texture(random.choice(patches), target_h, target_w)
    
    # 50% chance for at blende en anden patch ind svagt
    if random.random() > 0.5:
        secondary = tile_texture(random.choice(patches), target_h, target_w)
        alpha = random.uniform(0.1, 0.3)
        primary = primary * (1 - alpha) + secondary * alpha
    
    return primary

def blend_textures(img, strength):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    white_mask = np.clip((gray - 0.6) / 0.4, 0, 1)[:, :, np.newaxis]
    black_mask = np.clip((0.4 - gray) / 0.4, 0, 1)[:, :, np.newaxis]
    gray_mask  = np.clip(1.0 - white_mask - black_mask, 0, 1)

    wt = stitch_texture(white_patches, h, w)
    bt = stitch_texture(black_patches, h, w)
    gt = stitch_texture(gray_patches,  h, w)

    # Blacks får 1.5x strength relativt til resten
    combined = wt * white_mask * strength + bt * black_mask * (strength * 2.0) + gt * gray_mask * strength
    result = img.astype(np.float32) + combined
    return np.clip(result, 0, 255).astype(np.uint8)

def add_vignette(img):
    h, w = img.shape[:2]
    strength = random.uniform(0.0, 0.25)
    
    cx, cy = w / 2, h / 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    dist = dist / dist.max()
    
    vignette = 1 - dist * strength
    vignette = vignette[:, :, np.newaxis]
    
    result = img.astype(np.float32) * vignette
    return np.clip(result, 0, 255).astype(np.uint8)

def add_rotation(img):
    if random.random() > 0.5:
        return img
    angle = random.uniform(-2, 2)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def degrade(img_path, output_path):
    img = cv2.imread(str(img_path))

    grain_strength  = random.uniform(1.8, 2.5)
    blur_sigma      = random.uniform(0.3, 1.5)
    jpeg_quality_1  = random.randint(45, 80)
    jpeg_quality_2  = random.randint(60, 90)
    contrast_scale  = random.uniform(0.65, 0.82)
    shadow_lift     = random.uniform(20, 90)
    blacks_lift     = random.uniform(20, 50)

    if random.random() < 0.4:
        contrast_scale = random.uniform(0.20, 0.50)
        shadow_lift    = random.uniform(80, 100)
        blur_sigma     = random.uniform(0.8, 1.5)

def add_uneven_lighting(img):
    h, w = img.shape[:2]
    cx = random.randint(w//4, 3*w//4)
    cy = random.randint(h//4, 3*h//4)
    
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    dist = dist / dist.max()
    
    brightness = 1 + (1 - dist) * random.uniform(0.05, 0.20)
    brightness = brightness[:, :, np.newaxis]
    
    result = img.astype(np.float32) * brightness
    return np.clip(result, 0, 255).astype(np.uint8)

def degrade(img_path, output_path):
    img = cv2.imread(str(img_path))

    grain_strength  = random.uniform(1.8, 2.5)
    blur_sigma      = random.uniform(0.3, 1.5)
    jpeg_quality_1  = random.randint(45, 80)
    jpeg_quality_2  = random.randint(60, 90)
    contrast_scale  = random.uniform(0.65, 0.82)
    shadow_lift     = random.uniform(20, 70)
    blacks_lift     = random.uniform(20, 40)

    if random.random() < 0.4:
        contrast_scale = random.uniform(0.20, 0.50)
        shadow_lift    = random.uniform(80, 90)
        blur_sigma     = random.uniform(0.8, 1.5)

    # 1. Papirtekstur
    img = blend_textures(img, grain_strength)

    # 2. Generel kontrast + shadow lift
    img = img.astype(np.float32)
    img = img * contrast_scale + shadow_lift
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 3. Blacks lift
    img = img.astype(np.float32)
    dark_mask = img < 120
    img[dark_mask] = np.clip(img[dark_mask] + blacks_lift, 0, 255)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 4. Farvetoning
    img = img.astype(np.float32)
    img[:, :, 0] *= random.uniform(0.92, 0.98)
    img[:, :, 1] *= random.uniform(0.95, 0.99)
    img[:, :, 2] *= random.uniform(1.00, 1.03)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 5. Blur
    img = cv2.GaussianBlur(img, (0, 0), blur_sigma)

    # 6. Første JPEG runde
    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality_1])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # 7. Ujævn belysning
    img = add_uneven_lighting(img)

    # 8. Vignetting
    img = add_vignette(img)

    # 9. Anden JPEG runde
    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality_2])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    cv2.imwrite(str(output_path), img)


# --- Kør på alle digitals ---
digital_dir = Path("data/digitals")
output_dir  = Path("data/degraded")
output_dir.mkdir(exist_ok=True)

files = list(digital_dir.rglob("*.jpg"))
print(f"Fandt {len(files)} filer")

for img_path in files:
    degrade(img_path, output_dir / img_path.name)
    print(f"Done: {img_path.name}")

def validate(n=5):
    out = Path("data/validate")
    out.mkdir(exist_ok=True)
    paths = list(digital_dir.rglob("*.jpg"))[:n]
    for i, p in enumerate(paths):
        clean = cv2.imread(str(p))
        degraded_path = Path("data/degraded") / p.name
        if not degraded_path.exists():
            continue
        degraded = cv2.imread(str(degraded_path))
        h = min(clean.shape[0], degraded.shape[0])
        comparison = np.hstack([
            cv2.resize(clean,    (int(clean.shape[1]    * h / clean.shape[0]),    h)),
            cv2.resize(degraded, (int(degraded.shape[1] * h / degraded.shape[0]), h))
        ])
        cv2.imwrite(str(out / f"compare_{i}.png"), comparison)
        print(f"Saved compare_{i}.png")

validate()