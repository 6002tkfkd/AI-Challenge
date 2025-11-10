# merge_trains.py
import os, glob, shutil

ROOT = "/home/work/HAN_K/2025-csu-sw-ai-challenge/archive"
SOURCES = [
    f"{ROOT}/train",              # 원본
    # f"{ROOT}/train_patch",        # 패치
    f"{ROOT}/train_crack",        # copypaste
    f"{ROOT}/train_aug_strong",   # strong offline aug
    # 필요하면 f"{ROOT}/train_aug" 등 추가
]

DEST_IMG = f"{ROOT}/train_merged/images"
DEST_MSK = f"{ROOT}/train_merged/masks"
os.makedirs(DEST_IMG, exist_ok=True)
os.makedirs(DEST_MSK, exist_ok=True)

IMG_EXTS = ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG")
MSK_DIR_CAND = ("masks_png","masks")  # PNG 우선
MSK_EXTS = ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG")

def copy_all(src_img_dir, src_msk_dir, tag):
    # 이미지
    paths = []
    for e in IMG_EXTS: paths += glob.glob(os.path.join(src_img_dir, e))
    for p in paths:
        stem, ext = os.path.splitext(os.path.basename(p))
        dest = os.path.join(DEST_IMG, stem + ext)
        if os.path.exists(dest):  # 충돌 시 태그 부여
            dest = os.path.join(DEST_IMG, stem + f"_{tag}" + ext)
        shutil.copy2(p, dest)
    # 마스크
    mpaths = []
    if src_msk_dir and os.path.isdir(src_msk_dir):
        for e in MSK_EXTS: mpaths += glob.glob(os.path.join(src_msk_dir, e))
    for p in mpaths:
        stem, ext = os.path.splitext(os.path.basename(p))
        # 마스크는 PNG로 통일 저장
        dest = os.path.join(DEST_MSK, stem + ".png")
        if os.path.exists(dest):
            dest = os.path.join(DEST_MSK, stem + f"_{tag}.png")
        shutil.copy2(p, dest)

for s in SOURCES:
    imgd = os.path.join(s, "images")
    # masks_png 우선, 없으면 masks 사용
    mskd = None
    for cand in MSK_DIR_CAND:
        d = os.path.join(s, cand if cand in s else cand)  # train/masks_png or train/masks
        if os.path.isdir(d): mskd = d; break
    if not mskd:
        cand = os.path.join(s, "masks")
        if os.path.isdir(cand): mskd = cand
    tag = os.path.basename(s)
    if os.path.isdir(imgd):
        copy_all(imgd, mskd, tag)
        print(f"[OK] merged {s}")
    else:
        print(f"[SKIP] no images: {imgd}")

print("[DONE] merged into", DEST_IMG, "and", DEST_MSK)
