import os, glob, torch, random, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

# ===================== Augmentation =====================
class AugCompose:
    def __init__(self, ops): self.ops = ops
    def __call__(self, img, mask):
        for op in self.ops:
            img, mask = op(img, mask)
        return img, mask

class RandomHorizontalFlipPair:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            img, mask = TF.hflip(img), TF.hflip(mask)
        return img, mask

class RandomVerticalFlipPair:
    def __init__(self, p=0.1): self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            img, mask = TF.vflip(img), TF.vflip(mask)
        return img, mask

class RandomAffinePair:
    def __init__(self, degrees=5, translate=(0.05,0.05), scale=(0.9,1.1), shear=(-5,5)):
        self.degrees, self.translate, self.scale, self.shear = degrees, translate, scale, shear
    def __call__(self, img, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        tx = random.uniform(-self.translate[0], self.translate[0])
        ty = random.uniform(-self.translate[1], self.translate[1])
        sx = random.uniform(self.scale[0], self.scale[1])
        sy = sx
        shear = random.uniform(self.shear[0], self.shear[1])
        img  = TF.affine(img, angle=angle, translate=(int(tx*img.width), int(ty*img.height)),
                         scale=sx, shear=shear, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        mask = TF.affine(mask, angle=angle, translate=(int(tx*mask.width), int(ty*mask.height)),
                         scale=sy, shear=shear, interpolation=TF.InterpolationMode.NEAREST, fill=0)
        return img, mask

class RandomBrightnessContrast:
    def __init__(self, b=0.1, c=0.1): self.b, self.c = b, c
    def __call__(self, img, mask):
        if random.random() < 0.7:
            bf = 1.0 + random.uniform(-self.b, self.b)
            cf = 1.0 + random.uniform(-self.c, self.c)
            img = TF.adjust_brightness(img, bf)
            img = TF.adjust_contrast(img, cf)
        return img, mask

class RandomGaussianBlur:
    def __init__(self, p=0.15, kernel=3, sigma=(0.1, 0.8)):
        self.p, self.kernel, self.sigma = p, kernel, sigma
    def __call__(self, img, mask):
        if random.random() < self.p:
            s = random.uniform(self.sigma[0], self.sigma[1])
            img = TF.gaussian_blur(img, kernel_size=self.kernel, sigma=s)
        return img, mask

class RandomGaussianNoise:
    def __init__(self, p=0.15, std=0.02): self.p, self.std = p, std
    def __call__(self, img, mask):
        if random.random() < self.p:
            arr = np.array(img).astype(np.float32)/255.0
            arr = np.clip(arr + np.random.normal(0, self.std, arr.shape), 0, 1)
            img = Image.fromarray((arr*255).astype(np.uint8))
        return img, mask

# ===================== Extra augmentations for overfit_guard =====================
def _rand_perspective_points(w, h, distortion_scale=0.15):
    # 4개 코너를 distortion_scale 비율로 랜덤 이동
    dx = distortion_scale * w
    dy = distortion_scale * h
    pts = [
        (random.uniform(0, dx),             random.uniform(0, dy)),              # TL
        (w - random.uniform(0, dx),         random.uniform(0, dy)),              # TR
        (w - random.uniform(0, dx),         h - random.uniform(0, dy)),          # BR
        (random.uniform(0, dx),             h - random.uniform(0, dy)),          # BL
    ]
    start = [(0,0), (w,0), (w,h), (0,h)]
    end = pts
    return start, end

class RandomPerspectivePair:
    def __init__(self, p=0.3, distortion_scale=0.15):
        self.p = p
        self.distortion_scale = distortion_scale
    def __call__(self, img, mask):
        if random.random() < self.p:
            w, h = img.size
            startpoints, endpoints = _rand_perspective_points(w, h, self.distortion_scale)
            img = TF.perspective(img, startpoints, endpoints, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.perspective(mask, startpoints, endpoints, interpolation=TF.InterpolationMode.NEAREST)
        return img, mask

class RandomCutoutImageOnly:
    def __init__(self, p=0.25, holes=2, max_rel_size=0.12):
        self.p = p; self.holes = holes; self.max_rel_size = max_rel_size
    def __call__(self, img, mask):
        if random.random() < self.p:
            W, H = img.size
            arr = np.array(img, dtype=np.uint8)
            max_w = int(W * self.max_rel_size)
            max_h = int(H * self.max_rel_size)
            for _ in range(self.holes):
                cw = random.randint(max(2, max_w//2), max(2, max_w))
                ch = random.randint(max(2, max_h//2), max(2, max_h))
                x0 = random.randint(0, max(0, W - cw))
                y0 = random.randint(0, max(0, H - ch))
                # 주변 평균 밝기 + 약간의 노이즈로 자연스럽게
                local = arr[max(0,y0-3):y0+ch+3, max(0,x0-3):x0+cw+3]
                fill = int(np.clip(local.mean() + np.random.randn()*5, 0, 255))
                arr[y0:y0+ch, x0:x0+cw] = fill
            img = Image.fromarray(arr)
        return img, mask

# ------------------- augmentation builder -------------------
def build_train_augs(cfg):
    aug_cfg = cfg.get("augmentation", {})
    if not aug_cfg.get("enabled", True):
        print("[AUGMENTATION] disabled")
        return None

    preset = aug_cfg.get("preset", "").lower()
    if preset == "light":
        preset_vals = dict(rotate=3, brightness=0.05, contrast=0.05, noise_std=0.01, blur_p=0.10)
    elif preset == "strong":
        preset_vals = dict(rotate=10, brightness=0.20, contrast=0.20, noise_std=0.04, blur_p=0.25)
    elif preset == "overfit_guard":  # ★ 추가된 프리셋
        preset_vals = dict(
            rotate=8, brightness=0.20, contrast=0.20,
            noise_std=0.03, blur_p=0.25,
            flip_h=0.5, flip_v=0.1,
            translate=0.06, scale_min=0.88, scale_max=1.12, shear=6,
            persp_p=0.3, persp_distort=0.15,
            cutout_p=0.25, cutout_holes=2, cutout_max=0.12
        )
    else:
        preset_vals = dict(rotate=5, brightness=0.10, contrast=0.10, noise_std=0.02, blur_p=0.15)

    # 1) preset 반영 + 사용자 오버라이드 병합
    merged = {**preset_vals, **aug_cfg}

    # 2) 최종 셋
    effective = {
        "flip_h":     merged.get("flip_h", 0.5),
        "flip_v":     merged.get("flip_v", 0.1),
        "rotate":     merged.get("rotate", 5),
        "translate":  merged.get("translate", 0.05),
        "scale_min":  merged.get("scale_min", 0.9),
        "scale_max":  merged.get("scale_max", 1.1),
        "shear":      merged.get("shear", 5),
        "brightness": merged.get("brightness", 0.1),
        "contrast":   merged.get("contrast", 0.1),
        "blur_p":     merged.get("blur_p", 0.15),
        "blur_sigma": tuple(merged.get("blur_sigma", [0.1, 0.8])),
        "noise_p":    merged.get("noise_p", 0.15),
        "noise_std":  merged.get("noise_std", 0.02),
        "persp_p":    merged.get("persp_p", 0.0),
        "persp_distort": merged.get("persp_distort", 0.0),
        "cutout_p":   merged.get("cutout_p", 0.0),
        "cutout_holes": merged.get("cutout_holes", 0),
        "cutout_max": merged.get("cutout_max", 0.0),
    }

    # 보기 좋은 요약 로그
    kv = " | ".join([f"{k}={effective[k]}" for k in effective])
    print(f"[AUGMENTATION] preset={preset or 'medium'} | {kv}")

    # 3) 변환 생성
    return AugCompose([
        RandomHorizontalFlipPair(p=effective["flip_h"]),
        RandomVerticalFlipPair(p=effective["flip_v"]),
        RandomAffinePair(
            degrees=effective["rotate"],
            translate=(effective["translate"], effective["translate"]),
            scale=(effective["scale_min"], effective["scale_max"]),
            shear=(-effective["shear"], effective["shear"])
        ),
        RandomPerspectivePair(p=effective["persp_p"], distortion_scale=effective["persp_distort"]),
        RandomBrightnessContrast(b=effective["brightness"], c=effective["contrast"]),
        RandomGaussianBlur(p=effective["blur_p"], kernel=3, sigma=effective["blur_sigma"]),
        RandomGaussianNoise(p=effective["noise_p"], std=effective["noise_std"]),
        RandomCutoutImageOnly(p=effective["cutout_p"], holes=effective["cutout_holes"], max_rel_size=effective["cutout_max"]),
    ])


# ===================== Dataset (PNG/JPG 둘 다 지원) =====================
# ===================== Dataset (PNG/JPG 둘 다 지원) =====================
class CrackDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, size=(192,192), augment=None):
        self.root_dir = root_dir
        self.size = size
        self.augment = augment

        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks_png")
        if not os.path.isdir(self.mask_dir):
            self.mask_dir = os.path.join(root_dir, "masks")
        assert os.path.isdir(self.img_dir), f"images dir not found: {self.img_dir}"
        assert os.path.isdir(self.mask_dir), f"masks dir not found: {self.mask_dir}"

        # 이미지 확장자 다양성 지원
        img_paths = []
        for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            img_paths += glob.glob(os.path.join(self.img_dir, pat))
        img_paths = sorted(img_paths)

        # 같은 stem의 마스크를 png/jpg에서 자동 탐색
        self.samples = []
        for ip in img_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            mp = self._find_mask_by_stem(stem)
            if mp is not None:
                self.samples.append((ip, mp))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No (image, mask) pairs in {root_dir}. "
                f"Check extensions & folder names (masks_png/masks)."
            )
        print(f"[CrackDataset] {root_dir} -> {len(self.samples)} pairs")

    def _find_mask_by_stem(self, stem: str):
        for ext in (".png", ".PNG", ".jpg", ".jpeg", ".JPG", ".JPEG"):
            p = os.path.join(self.mask_dir, stem + ext)
            if os.path.isfile(p):
                return p
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ip, mp = self.samples[idx]
        img  = Image.open(ip).convert("L")
        mask = Image.open(mp).convert("L")

        # 항상 192x192
        img  = TF.resize(img,  self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        # JPG 마스크 노이즈 대비: 이진화(0/255)
        m_arr = np.array(mask, dtype=np.uint8)
        m_arr = ((m_arr > 127).astype(np.uint8) * 255)
        mask  = Image.fromarray(m_arr)

        if self.augment:
            img, mask = self.augment(img, mask)

        img_t  = torch.from_numpy(np.array(img,  dtype=np.float32) / 255.0).unsqueeze(0)  # [1,H,W]
        mask_t = torch.from_numpy((np.array(mask, dtype=np.uint8) > 0).astype(np.float32)).unsqueeze(0)
        return img_t, mask_t


# ===================== Losses =====================
class ContourWeightedLoss(nn.Module):
    """Emphasize boundary accuracy with weighted BCE + dual Dice terms."""
    def __init__(self, contour_weight=10.0):
        super().__init__()
        self.contour_weight = contour_weight

    def _extract_contour(self, masks):
        # Erode via -max_pool to highlight 1px boundaries.
        eroded = -F.max_pool2d(-masks, kernel_size=3, stride=1, padding=1)
        contour = masks - eroded
        return contour

    def forward(self, logits, targets):
        contour_map = self._extract_contour(targets)

        weight_map = torch.ones_like(targets)
        weight_map[targets > 0.5] = 3.0
        weight_map[contour_map > 0.5] = self.contour_weight

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weighted_bce = (bce * weight_map).mean()

        probs = torch.sigmoid(logits)

        contour_preds = probs * contour_map
        contour_targets = targets * contour_map
        contour_dice = (2 * (contour_preds * contour_targets).sum() + 1e-6) / (
            contour_preds.sum() + contour_targets.sum() + 1e-6
        )

        interior_mask = (targets > 0.5) & (contour_map < 0.5)
        interior_preds = probs * interior_mask.float()
        interior_targets = targets * interior_mask.float()
        interior_dice = (2 * (interior_preds * interior_targets).sum() + 1e-6) / (
            interior_preds.sum() + interior_targets.sum() + 1e-6
        )

        total_dice_loss = 1.0 - (0.7 * contour_dice + 0.3 * interior_dice)
        return weighted_bce + total_dice_loss

# ===================== Utility =====================
def set_seed_all(seed: int = 2025, verbose: bool = True):
    import os, random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # cuDNN 결정론 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # (선택) PyTorch 전역 결정론 강제 — 일부 연산에서 에러가 날 수 있음
    try:
        torch.use_deterministic_algorithms(True)
        # CUDA 결정론 이슈가 있으면 다음 환경변수 중 하나가 필요할 수 있음:
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 또는 ":4096:8"
    except Exception:
        pass

    if verbose:
        print(f"[seed] set to {seed}")

def binary_metrics(preds, targets, eps=1e-6):
    preds = preds.float(); targets = targets.float()
    tp = (preds*targets).sum(dim=(1,2,3))
    fp = (preds*(1-targets)).sum(dim=(1,2,3))
    fn = ((1-preds)*targets).sum(dim=(1,2,3))
    precision = (tp+eps)/(tp+fp+eps)
    recall    = (tp+eps)/(tp+fn+eps)
    f1 = (2*precision*recall+eps)/(precision+recall+eps)
    union = tp+fp+fn
    iou = (tp+eps)/(union+eps)
    return {"iou": iou.mean().item(), "f1": f1.mean().item()}

def estimate_pos_weight(loader, max_batches=10):
    total_pos, total_pix = 0, 0
    for i, (_, m) in enumerate(loader):
        if i >= max_batches: break
        total_pos += m.sum().item()
        total_pix += m.numel()
    pos = max(total_pos/(total_pix+1e-6), 1e-6)
    return (1.0 - pos) / pos
