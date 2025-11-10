# copypaste_crack.py (fixed, 192x192, auto mask ext, binarize)
import os, cv2, glob, numpy as np
from tqdm import tqdm

SRC_IMG = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train/images"
SRC_MSK = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train/masks"

DST_IMG = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train_crack/images"
DST_MSK = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive/train_crack/masks"
os.makedirs(DST_IMG, exist_ok=True); os.makedirs(DST_MSK, exist_ok=True)

SIZE = 192
COPIES_PER_BASE = 2          # 한 배경 이미지당 합성본 몇 개 만들지
MASK_EXTS = [".png",".jpg",".jpeg",".PNG",".JPG",".JPEG"]

def find_mask(stem: str):
    for ext in MASK_EXTS:
        p = os.path.join(SRC_MSK, stem + ext)
        if os.path.isfile(p):
            return p
    return None

def binarize(msk):
    if msk.dtype != np.uint8:
        msk = msk.astype(np.uint8)
    # JPG 마스크 노이즈 방지
    _, m = cv2.threshold(msk, 127, 255, cv2.THRESH_BINARY)
    return m

def extract_components(msk):
    nb, labels = cv2.connectedComponents((msk > 0).astype(np.uint8))
    comps = []
    for i in range(1, nb):
        comp = (labels == i).astype(np.uint8)
        if comp.sum() < 30:  # 너무 작은 컴포넌트 제거
            continue
        comps.append(comp)
    return comps

bgs = sorted(glob.glob(os.path.join(SRC_IMG, "*.jpg")))
if len(bgs) < 2:
    raise RuntimeError("need at least 2 images for copy-paste")

for ip in tqdm(bgs):
    stem = os.path.splitext(os.path.basename(ip))[0]
    mp = find_mask(stem)
    if mp is None:
        # 배경 쌍이 없으면 스킵
        continue

    bg = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
    bm = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if bg is None or bm is None:
        continue

    bg = cv2.resize(bg, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    bm = binarize(cv2.resize(bm, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST))

    for c in range(1, COPIES_PER_BASE + 1):
        out_i = bg.copy()
        out_m = bm.copy()

        # 랜덤 donor 선택 (자기 자신 제외)
        jp = np.random.choice([p for p in bgs if p != ip])
        jstem = os.path.splitext(os.path.basename(jp))[0]
        jmp = find_mask(jstem)
        if jmp is None:
            continue
        sm = cv2.imread(jmp, cv2.IMREAD_GRAYSCALE)
        if sm is None:
            continue
        sm = binarize(cv2.resize(sm, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST))

        # donor에서 컴포넌트 추출 후 랜덤 배치
        for comp in extract_components(sm):
            # 회전
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((SIZE/2, SIZE/2), angle, 1.0)
            comp_t = cv2.warpAffine(comp * 255, M, (SIZE, SIZE), flags=cv2.INTER_NEAREST)

            # 평행이동
            dx = np.random.randint(-10, 11); dy = np.random.randint(-10, 11)
            M2 = np.float32([[1, 0, dx], [0, 1, dy]])
            comp_t = cv2.warpAffine(comp_t, M2, (SIZE, SIZE), flags=cv2.INTER_NEAREST)

            mask = (comp_t > 0).astype(np.uint8)

            # 마스크 합성
            out_m = np.maximum(out_m, mask * 255)

            # 이미지 합성(균열 부분 어둡게)
            out_i[mask > 0] = (out_i[mask > 0] * 0.85).astype(np.uint8)

        cv2.imwrite(os.path.join(DST_IMG, f"{stem}_cp{c}.jpg"), out_i)
        cv2.imwrite(os.path.join(DST_MSK, f"{stem}_cp{c}.png"), out_m)
