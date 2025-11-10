# infer.py
import os, glob, yaml, torch, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from models.hrsegnet_b16_cbam import HrSegNetB16CBAM as Net

# ---- morphology helpers ----
def _one_morph_step(pred: np.ndarray, op: str, k: int = 3, iters: int = 1, shape: str = "rect") -> np.ndarray:
    """
    pred: (H,W) uint8 {0,1}
    op : "none"|"e"|"d"|"o"|"c"|"erode"|"dilate"|"open"|"close"
    """
    if not op or op == "none" or k <= 1 or iters <= 0:
        return pred

    op = {"e": "erode", "d": "dilate", "o": "open", "c": "close"}.get(op, op)
    m = (pred.astype(np.uint8) * 255)

    try:
        import cv2
        if shape == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        elif shape == "cross":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        if op == "erode":
            m = cv2.erode(m, kernel, iterations=iters)
        elif op == "dilate":
            m = cv2.dilate(m, kernel, iterations=iters)
        elif op == "open":
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=iters)
        elif op == "close":
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=iters)
        return (m > 127).astype(np.uint8)

    except Exception:
        # OpenCV가 없으면 scipy로 대체
        try:
            import scipy.ndimage as ndi
        except Exception:
            return pred

        selem = np.ones((k, k), dtype=bool)
        mb = pred.astype(bool)
        if op == "erode":
            for _ in range(iters): mb = ndi.binary_erosion(mb, structure=selem)
        elif op == "dilate":
            for _ in range(iters): mb = ndi.binary_dilation(mb, structure=selem)
        elif op == "open":
            for _ in range(iters): mb = ndi.binary_opening(mb, structure=selem)
        elif op == "close":
            for _ in range(iters): mb = ndi.binary_closing(mb, structure=selem)
        return mb.astype(np.uint8)

def apply_morphology_pipeline(pred: np.ndarray, ops: str = "none", k: int = 3, iters: int = 1, shape: str = "rect") -> np.ndarray:
    """
    ops 예시:
      "none"
      "open,close"
      "e,d,close" (축약 허용)
    """
    if not ops or ops.lower() == "none":
        return pred
    out = pred
    for op in [t.strip().lower() for t in ops.split(",") if t.strip()]:
        out = _one_morph_step(out, op, k=k, iters=iters, shape=shape)
    return out

# -------- RLE --------
def rle_encode(mask: np.ndarray) -> str:
    pixels = mask.flatten(order="C")
    ones = np.where(pixels == 1)[0] + 1
    if len(ones) == 0: return ""
    runs = []; prev = -2
    for idx in ones:
        if idx > prev + 1:
            runs.extend((idx, 0))
        runs[-1] += 1
        prev = idx
    return " ".join(map(str, runs))

def load_best_threshold(output_dir, default=0.5):
    p = os.path.join(output_dir, "best_threshold.txt")
    if os.path.exists(p):
        try:
            return float(open(p).read().strip())
        except:
            return default
    return default

def _measure_mmacs_with_thop(model, in_ch, H, W, device):
    """가능하면 thop으로 per-pass MACs 실측(MMAC)."""
    try:
        from thop import profile
    except ImportError:
        return None

    class _MainHead(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):  return self.m(x)[0]  # [B,1,H,W] logits

    wrapper = _MainHead(model).to(device).eval()
    dummy = torch.randn(1, in_ch, H, W, device=device)
    with torch.inference_mode():
        macs, _ = profile(wrapper, inputs=(dummy,), verbose=False)
    return macs / 1e6  # -> MMAC

def load_post_params(output_dir):
    """
    post_params.txt:
      thr=0.53
      min_area=12
      alpha=1.10
      beta=0.00
    없으면 None
    """
    p = os.path.join(output_dir, "post_params.txt")
    if not os.path.exists(p): return None
    vals = {}
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line: continue
            k, v = line.split("=", 1)
            k = k.strip().lower(); v = v.strip()
            try:
                vals[k] = int(v) if k == "min_area" else float(v)
            except:
                pass
    for k in ["thr", "min_area", "alpha", "beta"]:
        if k not in vals: return None
    return vals

def load_ckpt_smart(output_dir, device):
    for name in ("best_avg.pt", "best.pt"):
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            return torch.load(p, map_location=device), name
    raise FileNotFoundError("No checkpoint found: best_avg.pt or best.pt")

# -------- postproc: remove small comps --------
def remove_small(pred: np.ndarray, min_area: int = 0) -> np.ndarray:
    """pred: (H,W) uint8 {0,1} — min_area 미만의 4-연결 성분 제거"""
    if min_area <= 0: return pred
    try:
        import cv2
        n, lab = cv2.connectedComponents(pred.astype(np.uint8), connectivity=4)
        out = np.zeros_like(pred, dtype=np.uint8)
        for k in range(1, n):
            comp = (lab == k)
            if comp.sum() >= min_area:
                out[comp] = 1
        return out
    except Exception:
        # 느린 파이썬 BFS fallback
        H, W = pred.shape
        visited = np.zeros_like(pred, dtype=bool)
        out = pred.copy()
        for y in range(H):
            for x in range(W):
                if pred[y, x] and not visited[y, x]:
                    q = [(y, x)]; comp = [(y, x)]; visited[y, x] = True
                    while q:
                        cy, cx = q.pop()
                        for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                            if 0 <= ny < H and 0 <= nx < W and pred[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                q.append((ny, nx)); comp.append((ny, nx))
                    if len(comp) < min_area:
                        for (yy, xx) in comp: out[yy, xx] = 0
        return out

# -------- TTA helpers --------
def _rot90_img(t):   # +90deg
    return torch.flip(t.transpose(-1, -2), dims=[-1])
def _rot270_img(t):  # -90deg
    return torch.flip(t.transpose(-1, -2), dims=[-2])

@torch.no_grad()
def infer_logits_tta(model, ten, tta="none"):
    """returns logits (H,W); model outputs logits; we TTA-average in logit space; ten: (1,1,H,W)"""
    l0 = model(ten)[0][0, 0]  # (H,W)
    if tta == "none":
        return l0
    elif tta == "hflip":      # 2-pass
        l1 = model(torch.flip(ten, dims=[-1]))[0][0, 0]
        l1 = torch.flip(l1, dims=[-1])
        return 0.5 * (l0 + l1)
    elif tta == "hvflip":     # 3-pass
        l1 = model(torch.flip(ten, dims=[-1]))[0][0, 0]
        l1 = torch.flip(l1, dims=[-1])
        l2 = model(torch.flip(ten, dims=[-2]))[0][0, 0]
        l2 = torch.flip(l2, dims=[-2])
        return (l0 + l1 + l2) / 3.0
    elif tta == "rot180":     # 2-pass
        t180 = torch.flip(ten, dims=[-1, -2])
        l1 = model(t180)[0][0, 0]
        l1 = torch.flip(l1, dims=[-1, -2])
        return 0.5 * (l0 + l1)
    elif tta == "rot90":      # 3-pass: 0/90/270
        t90  = _rot90_img(ten)
        l90  = model(t90)[0][0, 0]
        l90  = _rot270_img(l90.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        t270 = _rot270_img(ten)
        l270 = model(t270)[0][0, 0]
        l270 = _rot90_img(l270.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        return (l0 + l90 + l270) / 3.0
    else:
        return l0

def _passes_for_tta(tta): return {"none":1, "hflip":2, "hvflip":3, "rot180":2, "rot90":3}.get(tta, 1)
def _gather_test_images(test_dir: str):
    paths = []
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths += glob.glob(os.path.join(test_dir, pat))
    return sorted(paths)

def main():
    cfg = yaml.safe_load(open("config.yaml", "r"))
    data_dir   = os.environ.get("DATA_DIR",   cfg["data_dir"])
    output_dir = os.environ.get("OUTPUT_DIR", cfg["output_dir"])

    # ----- 임계값/옵션 로드 -----
    # 1) post_params.txt 우선, 없으면 best_threshold.txt + ENV
    post = load_post_params(output_dir)
    if post is not None:
        thr      = float(post["thr"])
        min_area = int(post["min_area"])
        alpha    = float(post["alpha"])
        beta     = float(post["beta"])
    else:
        thr      = load_best_threshold(output_dir, default=0.5)
        min_area = int(os.environ.get("MIN_AREA", "0"))
        alpha    = float(os.environ.get("ALPHA", "1.0"))
        beta     = float(os.environ.get("BETA",  "0.0"))

    tta_mode    = os.environ.get("TTA",  "hvflip")
    temperature = float(os.environ.get("TEMP", "1.0"))

    # ----- 모델/체크포인트 -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(
        in_channels=cfg["in_channels"],
        base=cfg["base_channels"],
        num_classes=cfg["num_classes"],
        cbam_ratio=cfg.get("cbam_ratio", 8),
        cbam_kernel=cfg.get("cbam_kernel", 7)
    ).to(device)
    ckpt, chosen = load_ckpt_smart(output_dir, device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # ----- MMAC 실측(thop) 또는 ENV 추정 -----
    H = W = int(cfg["img_size"])
    measured_mmac = None
    if os.environ.get("CHECK_MACS", "1") == "1":
        measured_mmac = _measure_mmacs_with_thop(model, cfg["in_channels"], H, W, device)
    if measured_mmac is not None:
        per_pass_mmac = measured_mmac; mmac_src = "measured(thop)"
    else:
        per_pass_mmac = float(os.environ.get("MODEL_MMAC", "150")); mmac_src = "env MODEL_MMAC"

    passes     = _passes_for_tta(tta_mode)
    total_mmac = per_pass_mmac * passes
    print(f"[MACs] per-pass={per_pass_mmac:.1f} MMAC ({mmac_src}), TTA passes={passes}, total≈{total_mmac:.1f} MMAC")
    if total_mmac > 500:
        raise RuntimeError(f"MMAC budget exceeded: {passes} × {per_pass_mmac:.1f} = {total_mmac:.1f} > 500")

    # ----- Morphology 옵션 (파이프라인) -----
    # 우선순위: MORPH_OPS(여러 개) > MORPH_OP(단일)
    morph_ops   = os.environ.get("MORPH_OPS", os.environ.get("MORPH_OP", "none")).lower()
    morph_k     = int(os.environ.get("MORPH_K", "3"))
    morph_iters = int(os.environ.get("MORPH_ITERS", "1"))
    morph_shape = os.environ.get("MORPH_SHAPE", "rect").lower()

    print(f"[INFO] ckpt={chosen} | TTA={tta_mode} TEMP={temperature} | thr={thr} "
          f"min_area={min_area} alpha={alpha} beta={beta} | "
          f"MORPH ops='{morph_ops}' k={morph_k} iters={morph_iters} shape={morph_shape}")

    # ----- 테스트 이미지 수집(JPG/PNG) -----
    test_dir = os.path.join(data_dir, "test", "images")
    test_imgs = _gather_test_images(test_dir)
    if len(test_imgs) == 0:
        raise FileNotFoundError(f"No test images found under: {test_dir}")

    ids, rles = [], []
    with torch.inference_mode():
        for path in tqdm(test_imgs, total=len(test_imgs), desc="Infer [test]"):
            img_id = os.path.splitext(os.path.basename(path))[0]
            img = Image.open(path).convert("L").resize((W, H))
            arr = np.array(img, dtype=np.float32) / 255.0
            ten = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

            logit = infer_logits_tta(model, ten, tta=tta_mode)          # (H,W)
            prob  = torch.sigmoid((alpha * logit + beta) * temperature).cpu().numpy()
            pred  = (prob > thr).astype(np.uint8)                        # (H,W) {0,1}
            pred  = remove_small(pred, min_area=min_area)
            pred  = apply_morphology_pipeline(pred, ops=morph_ops, k=morph_k, iters=morph_iters, shape=morph_shape)

            ids.append(img_id)
            rles.append(rle_encode(pred))

    df = pd.DataFrame({"image_id": ids, "rle": rles})
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "submission.csv")
    df.to_csv(out_csv, index=False)
    print(f"[OK] saved: {out_csv} ({len(df)} rows) | passes={passes}, est_MMAC≈{total_mmac:.0f}")

if __name__ == "__main__":
    main()
