# augment_offline_balanced.py
import os, glob, cv2, numpy as np
from tqdm import tqdm
import albumentations as A

SRC_IMG = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train/images"
SRC_MSK = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train/masks"

DST_IMG = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train_aug/images"
DST_MSK = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train_aug/masks"

K = 3
SIZE = 192

os.makedirs(DST_IMG, exist_ok=True); os.makedirs(DST_MSK, exist_ok=True)

aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.22),
    A.Affine(scale=(0.88,1.15), translate_percent=(0.0,0.08),
             rotate=(-12,12), shear=(-8,8),
             border_mode=cv2.BORDER_REFLECT_101, fit_output=False, p=0.9),
    A.RandomBrightnessContrast(0.23, 0.22, p=0.7),
    A.GaussianBlur(blur_limit=(3,5), p=0.30),
    A.GaussNoise(p=0.25),
    A.Perspective(scale=(0.05,0.15), p=0.30),
    A.CoarseDropout(holes_number_range=(1,2),
                    hole_height_range=(int(SIZE*0.06), int(SIZE*0.12)),
                    hole_width_range=(int(SIZE*0.06), int(SIZE*0.12)),
                    fill_value=0, p=0.25),
    A.Resize(SIZE, SIZE)
])

imgs = sorted(glob.glob(os.path.join(SRC_IMG, "*.jpg")))
skip=0
for ip in tqdm(imgs):
    stem = os.path.splitext(os.path.basename(ip))[0]
    mp = os.path.join(SRC_MSK, stem + ".png")
    if not os.path.isfile(mp):
        skip += 1
        continue
    img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        skip += 1
        continue

    for k in range(1, K+1):
        out = aug(image=img, mask=msk)
        ai, am = out["image"], out["mask"]
        if am.dtype != np.uint8: am = am.astype(np.uint8)
        _, am = cv2.threshold(am, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(DST_IMG, f"{stem}_a{k}.jpg"), ai)
        cv2.imwrite(os.path.join(DST_MSK, f"{stem}_a{k}.png"), am)
print(f"[INFO] done. skipped={skip}")
