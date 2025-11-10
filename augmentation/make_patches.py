# make_patches_smart.py
import os, cv2, glob, numpy as np
from tqdm import tqdm

SRC_IMG = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train/images"
SRC_MSK = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train/masks"  # or masks_png
DST_IMG = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train_patch/images"
DST_MSK = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train_patch/masks"
os.makedirs(DST_IMG, exist_ok=True); os.makedirs(DST_MSK, exist_ok=True)

PATCH, STRIDE = 192, 96
POS_RATIO = 0.6
MASK_EXTS = (".png",".jpg",".jpeg",".PNG",".JPG",".JPEG")

def find_mask(stem):
    for ext in MASK_EXTS:
        p = os.path.join(SRC_MSK, stem+ext)
        if os.path.isfile(p):
            return p
    return None

def binarize(m):
    if m.dtype != np.uint8: m = m.astype(np.uint8)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m

total, skipped_same = 0, 0
for ip in tqdm(sorted(glob.glob(os.path.join(SRC_IMG, "*.jpg")))):
    stem = os.path.splitext(os.path.basename(ip))[0]
    mp = find_mask(stem)
    if mp is None: continue
    img = cv2.imread(ip, 0); msk = cv2.imread(mp, 0)
    if img is None or msk is None: continue
    H, W = img.shape

    # 이미 192x192면 스킵(복사 안 함)
    if H == PATCH and W == PATCH:
        skipped_same += 1
        continue

    pid = 0
    for y in range(0, max(1, H-PATCH+1), STRIDE):
        for x in range(0, max(1, W-PATCH+1), STRIDE):
            pi, pm = img[y:y+PATCH, x:x+PATCH], msk[y:y+PATCH, x:x+PATCH]
            # 가장자리 패딩
            if pi.shape != (PATCH, PATCH):
                pad_i = np.zeros((PATCH, PATCH), np.uint8)
                pad_m = np.zeros((PATCH, PATCH), np.uint8)
                ph, pw = pi.shape
                pad_i[:ph, :pw] = pi
                pad_m[:ph, :pw] = pm
                pi, pm = pad_i, pad_m
