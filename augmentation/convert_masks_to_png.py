# convert_masks_to_png.py
import os, glob, cv2, numpy as np

SRC = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train/masks"  # JPG가 있는 폴더
DST = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train/masks_png"  # 출력 폴더

os.makedirs(DST, exist_ok=True)
paths = sorted(glob.glob(os.path.join(SRC, "*.jpg"))) + sorted(glob.glob(os.path.join(SRC, "*.jpeg")))
cnt=0
for p in paths:
    stem = os.path.splitext(os.path.basename(p))[0]
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        print("[WARN] fail read:", p); 
        continue
    # 이진화(임계 127), 출력은 0/255
    _, mb = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(DST, stem + ".png"), mb)
    cnt += 1
print(f"[OK] converted {cnt} masks to PNG")
