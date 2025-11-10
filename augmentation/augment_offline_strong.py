# augment_offline_strong.py
import os, glob, cv2, numpy as np
from tqdm import tqdm
import albumentations as A

SRC_IMG = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train/images"
SRC_MSK = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train/masks"

DST_IMG = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train_aug_strong/images"
DST_MSK = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train_aug_strong/masks"

K = 5
SIZE = 192
MASK_EXTS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]  # ★ 자동 탐색

os.makedirs(DST_IMG, exist_ok=True)
os.makedirs(DST_MSK, exist_ok=True)

def find_mask(stem: str):
    for ext in MASK_EXTS:
        p = os.path.join(SRC_MSK, stem + ext)
        if os.path.isfile(p):
            return p
    return None

aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.35),
    A.Affine(scale=(0.80,1.25), translate_percent=(0.00,0.12),
             rotate=(-25,25), shear=(-12,12),
             border_mode=cv2.BORDER_REFLECT_101, fit_output=False, p=0.95),
    A.RandomBrightnessContrast(0.30, 0.30, p=0.75),
    A.GaussianBlur(blur_limit=(3,7), p=0.40),
    A.GaussNoise(p=0.35),                      # 2.0.8 호환
    A.ElasticTransform(alpha=25, sigma=5, p=0.30),
    A.Perspective(scale=(0.08,0.18), p=0.40),
    A.CoarseDropout(holes_number_range=(2,4),
                    hole_height_range=(int(SIZE*0.10), int(SIZE*0.22)),
                    hole_width_range=(int(SIZE*0.10), int(SIZE*0.22)),
                    fill_value=0, p=0.45),
    A.Resize(SIZE, SIZE),
])

imgs = sorted(glob.glob(os.path.join(SRC_IMG, "*.jpg")))
skipped = 0
for ip in tqdm(imgs):
    stem = os.path.splitext(os.path.basename(ip))[0]
    mp = find_mask(stem)
    if mp is None:
        skipped += 1
        # print(f"[WARN] mask not found: {stem}")
        continue

    img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        skipped += 1
        continue

    for k in range(1, K+1):
        out = aug(image=img, mask=msk)
        ai, am = out["image"], out["mask"]

        # ★ JPG 마스크 노이즈 방지: 이진화 후 PNG 저장
        if am.dtype != np.uint8: am = am.astype(np.uint8)
        _, am = cv2.threshold(am, 127, 255, cv2.THRESH_BINARY)

        cv2.imwrite(os.path.join(DST_IMG, f"{stem}_s{k}.jpg"), ai)
        cv2.imwrite(os.path.join(DST_MSK, f"{stem}_s{k}.png"), am)

print(f"[INFO] done. total={len(imgs)}, skipped(no-mask or read-fail)={skipped}")
