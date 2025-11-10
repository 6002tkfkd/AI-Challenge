# train.py
import os, time, yaml, cv2, numpy as np, torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from models.hrsegnet_b16_cbam import HrSegNetB16CBAM as Net
from utils import (
    CrackDataset,
    set_seed_all,
    binary_metrics,
    build_train_augs,
    ContourWeightedLoss,
)

# ---------------- EMA ----------------
class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.ema = self._clone_model(model)
        self.ema.eval()
        self.decay = decay
        if device is not None:
            self.ema.to(device)

    @torch.no_grad()
    def _clone_model(self, model):
        import copy
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd and v.dtype.is_floating_point:
                v.copy_(v * d + msd[k].detach() * (1.0 - d))

# ---------------- checkpoint avg ----------------
def average_checkpoints(ckpt_paths, device):
    avg = None; n=0
    for p in ckpt_paths:
        sd = torch.load(p, map_location=device)["model"]
        if avg is None:
            avg = {k: v.clone().float() for k, v in sd.items()}
        else:
            for k in avg.keys():
                avg[k] += sd[k].float()
        n += 1
    for k in avg.keys():
        avg[k] /= max(n, 1)
    return avg

# ---------------- TTA helpers (validation) ----------------
@torch.no_grad()
def _tta_logits(model_like, imgs, tta):
    logits0 = model_like(imgs)[0]
    if tta == "hflip":
        l1 = model_like(torch.flip(imgs, dims=[-1]))[0]; l1 = torch.flip(l1, dims=[-1])
        return 0.5 * (logits0 + l1)
    elif tta == "hvflip":
        l1 = model_like(torch.flip(imgs, dims=[-1]))[0]; l1 = torch.flip(l1, dims=[-1])
        l2 = model_like(torch.flip(imgs, dims=[-2]))[0]; l2 = torch.flip(l2, dims=[-2])
        return (logits0 + l1 + l2) / 3.0
    else:
        return logits0

# 같은 로직을 (alpha/beta 튜닝용) 재사용하기 위한 helper
@torch.no_grad()
def _tta_logits_like(model_like, imgs, tta):
    return _tta_logits(model_like, imgs, tta)

# ---------------- threshold/min_area 공동 탐색 ----------------
@torch.no_grad()
def search_thr_area(model_like, val_loader, device, tta="hvflip", temperature=1.0,
                    center=0.5, radius=0.12, thr_step=0.01,
                    areas=(0,4,8,12,16,24)):
    ths = np.arange(max(0, center-radius), min(1, center+radius)+1e-9, thr_step)
    best = (center, 0, -1.0)  # thr, min_area, score

    for ma in areas:
        f1s_acc = {thr: [] for thr in ths}
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = _tta_logits(model_like, imgs, tta)
            probs  = torch.sigmoid(logits * temperature).cpu().numpy()
            gt     = masks.cpu().numpy().astype(np.uint8)

            # 배치 내에서 여러 thr를 한 번에 평가
            for thr in ths:
                pred = (probs > thr).astype(np.uint8)

                # 작은 컴포넌트 제거(4-연결)
                if ma > 0:
                    for i in range(pred.shape[0]):
                        p = pred[i,0]
                        n, lab = cv2.connectedComponents(p, connectivity=4)
                        keep = np.zeros_like(p)
                        for k in range(1, n):
                            comp = (lab == k)
                            if comp.sum() >= ma:
                                keep[comp] = 1
                        pred[i,0] = keep

                m = binary_metrics(torch.from_numpy(pred), torch.from_numpy(gt))
                f1s_acc[thr].append(m["f1"])

        # thr별 평균 f1 비교
        for thr in ths:
            score = float(np.mean(f1s_acc[thr])) if len(f1s_acc[thr]) > 0 else -1.0
            if score > best[2]:
                best = (float(thr), int(ma), score)
    return best  # (thr, min_area, f1)

# ---------------- alpha/beta (logit scaling & shift) 탐색 ----------------
@torch.no_grad()
def tune_alpha_beta(model_like, val_loader, device, tta="hvflip",
                    alphas=np.linspace(0.5, 2.0, 16), betas=np.linspace(-1.0, 1.0, 21),
                    thr=0.5):
    best = (1.0, 0.0, -1.0)  # alpha, beta, f1
    for a in alphas:
        for b in betas:
            f1s = []
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = _tta_logits_like(model_like, imgs, tta)
                probs  = torch.sigmoid(a * logits + b)
                pred   = (probs > thr).float()
                m = binary_metrics(pred, masks)
                f1s.append(m["f1"])
            score = float(np.mean(f1s))
            if score > best[2]:
                best = (float(a), float(b), score)
    return best

# ---------------- coarse/fine threshold search ----------------
@torch.no_grad()
def find_best_threshold_coarse(model_like, val_loader, device, tta="hvflip", temperature=1.0):
    model_like.eval()
    ths = np.linspace(0.2, 0.8, 13)  # 0.20~0.80, step 0.05
    best = (0.5, -1.0)
    for thr in ths:
        f1s=[]
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = _tta_logits(model_like, imgs, tta)
            pred = (torch.sigmoid(logits * temperature) > thr).float()
            m = binary_metrics(pred, masks)
            f1s.append(m["f1"])
        score = float(np.mean(f1s))
        if score > best[1]:
            best = (float(thr), score)
    return best

@torch.no_grad()
def find_best_threshold_fine(model_like, val_loader, device, tta="hvflip", temperature=1.0,
                             center=0.5, radius=0.15, step=0.01):
    model_like.eval()
    lo = max(0.0, center - radius); hi = min(1.0, center + radius)
    ths = np.arange(lo, hi + 1e-6, step)
    best = (center, -1.0)
    for thr in ths:
        f1s=[]
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = _tta_logits(model_like, imgs, tta)
            pred = (torch.sigmoid(logits * temperature) > thr).float()
            m = binary_metrics(pred, masks)
            f1s.append(m["f1"])
        score = float(np.mean(f1s))
        if score > best[1]:
            best = (float(thr), score)
    return best

def load_best_threshold(output_dir, default=0.5):
    p = os.path.join(output_dir, "best_threshold.txt")
    if os.path.exists(p):
        try:
            return float(open(p).read().strip())
        except:
            return default
    return default

# ---------------- main ----------------
def main():
    cfg = yaml.safe_load(open("config.yaml", "r"))
    set_seed_all(cfg.get("seed", 2025))  # 결정론 강제는 켜지 않음(속도/호환 이슈 방지)

    data_dir   = os.environ.get("DATA_DIR",   cfg["data_dir"])
    output_dir = os.environ.get("OUTPUT_DIR", cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    H = W = int(cfg["img_size"])
    train_set = CrackDataset(os.path.join(data_dir, "train"), size=(H,W), augment=build_train_augs(cfg))
    val_set   = CrackDataset(os.path.join(data_dir, "val"),   size=(H,W), augment=None)

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(
        in_channels=cfg["in_channels"],
        base=cfg["base_channels"],
        num_classes=cfg["num_classes"],
        cbam_ratio=cfg.get("cbam_ratio", 8),
        cbam_kernel=cfg.get("cbam_kernel", 7)
    ).to(device)

    # EMA
    ema = ModelEMA(model, decay=0.999, device=device)

    # loss
    criterion = ContourWeightedLoss(contour_weight=float(cfg.get("contour_weight", 10.0)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    use_amp = bool(cfg["amp"])
    scaler = GradScaler(device="cuda", enabled=use_amp)

    # TTA / Temperature (검증/탐색에서 사용)
    tta_mode    = os.environ.get("TTA", "hvflip")    # "none"|"hflip"|"hvflip"
    temperature = float(os.environ.get("TEMP", "1.0"))

    best_val_f1 = -1.0
    best_path   = os.path.join(output_dir, "best.pt")  # EMA 기준 best
    topk = 3
    best_ckpts = []
    global_step = 0
    t0 = time.time()

    # --- Early Stopping 설정 ---
    es_cfg      = cfg.get("early_stopping", {})
    es_on       = bool(es_cfg.get("enabled", True))
    es_monitor  = es_cfg.get("monitor", "val_f1")       # "val_f1" or "val_iou"
    es_mode_max = (es_cfg.get("mode", "max").lower() == "max")
    es_patience = int(es_cfg.get("patience", 12))
    es_min_delta= float(es_cfg.get("min_delta", 5e-4))
    es_warmup   = int(es_cfg.get("warmup_epochs", 5))

    no_improve_epochs = 0
    best_score = -float("inf") if es_mode_max else float("inf")

    # ----------------- Epoch Loop -----------------
    for epoch in range(1, cfg["epochs"]+1):
        model.train(); ep_loss=0.0
        aux = cfg["aux_init"] if epoch <= 5 else cfg["aux_late"]

        # --- Training progress bar ---
        pbar = tqdm(train_loader, total=len(train_loader),
                    desc=f"Epoch {epoch:03d}/{cfg['epochs']} [train]",
                    ncols=100, leave=False)

        for imgs, masks in pbar:
            global_step += 1
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                outs = model(imgs)  # [main, aux1, aux2]
                main = outs[0]
                loss = aux[0]*criterion(main, masks)
                if len(outs)>1: loss += aux[1]*criterion(outs[1], masks)
                if len(outs)>2: loss += aux[2]*criterion(outs[2], masks)

            scaler.scale(loss).backward()
            # (선택) gradient clipping
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            ep_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # -------- Validation --------
        model.eval(); ious=[]; f1s=[]
        vbar = tqdm(val_loader, total=len(val_loader),
                    desc=f"Epoch {epoch:03d}/{cfg['epochs']} [valid]",
                    ncols=100, leave=False)
        with torch.no_grad():
            for imgs, masks in vbar:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = _tta_logits(ema.ema, imgs, tta_mode)
                pred = (torch.sigmoid(logits * temperature) > 0.5).float()
                m = binary_metrics(pred, masks)
                ious.append(m["iou"]); f1s.append(m["f1"])
                vbar.set_postfix(iou=f"{m['iou']:.4f}", f1=f"{m['f1']:.4f}")

        val_iou, val_f1 = float(np.mean(ious)), float(np.mean(f1s))
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:03d}/{cfg['epochs']}] "
            f"lr={cur_lr:.6f} | loss={ep_loss/max(1,len(train_loader)):.4f} | "
            f"Val IoU={val_iou:.4f} | Val F1={val_f1:.4f} | "
            f"time={time.time()-t0:.1f}s",
            flush=True,
        )

        # EMA 기준 best 저장
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epoch_path = os.path.join(output_dir, f"best_e{epoch:03d}_{val_f1:.4f}.pt")
            torch.save({"model": ema.ema.state_dict(), "cfg": cfg}, epoch_path)
            best_ckpts.append(epoch_path)
            torch.save({"model": ema.ema.state_dict(), "cfg": cfg}, best_path)
            print(f"  ↳ saved {epoch_path} (EMA, F1={best_val_f1:.4f})", flush=True)
            if len(best_ckpts) > topk:
                old = best_ckpts.pop(0)
                try:
                    if os.path.exists(old): os.remove(old)
                except Exception:
                    pass

        # -------- (선택) 주기적 coarse 탐색 --------
        if epoch % 5 == 0:
            coarse_thr, coarse_f1 = find_best_threshold_coarse(
                ema.ema, val_loader, device, tta=tta_mode, temperature=temperature
            )
            with open(os.path.join(output_dir, "best_threshold.txt"), "w") as f:
                f.write(f"{coarse_thr:.3f}\n")
            print(f"  ↳ updated coarse threshold: {coarse_thr:.3f} (F1={coarse_f1:.4f})", flush=True)

        # ------- Early Stopping 체크 -------
        cur_score = val_f1 if es_monitor == "val_f1" else val_iou
        improved = (cur_score > best_score + es_min_delta) if es_mode_max else (cur_score < best_score - es_min_delta)
        if improved:
            best_score = cur_score
            no_improve_epochs = 0
        else:
            if epoch >= es_warmup and es_on:
                no_improve_epochs += 1
                print(f"  ↳ early-stop counter = {no_improve_epochs}/{es_patience}")
        if es_on and epoch >= es_warmup and no_improve_epochs >= es_patience:
            print(f"[EARLY STOP] '{es_monitor}' no improvement for {es_patience} epochs (best={best_score:.4f}).")
            break

    # -------- top-k 평균 체크포인트 저장 --------
    if len(best_ckpts) >= 1:
        avg_sd = average_checkpoints(best_ckpts, device)
        avg_path = os.path.join(output_dir, "best_avg.pt")
        torch.save({"model": avg_sd, "cfg": cfg}, avg_path)
        print(f"[OK] Saved averaged checkpoint: {avg_path} from {len(best_ckpts)} files")

    # -------- 종료 후 fine 탐색 --------
    # 우선순위: best_avg.pt → best.pt
    model_like = model
    if os.path.exists(os.path.join(output_dir, "best_avg.pt")):
        model.load_state_dict(torch.load(os.path.join(output_dir, "best_avg.pt"), map_location=device)["model"], strict=True)
        model_like = model
    elif os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["model"], strict=True)
        model_like = model
    else:
        print("[WARN] No best checkpoint found for threshold search.")

    # coarse 결과 있으면 그 근처에서 정밀 탐색
    if cfg.get("threshold_search", True):
        coarse = load_best_threshold(output_dir, default=0.5)
        fine_thr, fine_f1 = find_best_threshold_fine(
            model_like, val_loader, device, tta=tta_mode, temperature=temperature,
            center=coarse, radius=0.15, step=0.01
        )
        with open(os.path.join(output_dir, "best_threshold.txt"), "w") as f:
            f.write(f"{fine_thr:.3f}\n")
        print(f"[OK] refined threshold: {fine_thr:.3f} (F1={fine_f1:.4f})")

    # -------- (검증셋에서) post-params 튜닝 & 저장 --------
    model_like.eval()
    # thr & min_area 공동탐색 (best_threshold 중심으로 한 번 더 미세 탐색)
    center_thr = load_best_threshold(output_dir, default=0.5)
    best_thr, best_min_area, best_f1 = search_thr_area(
        model_like, val_loader, device,
        tta=tta_mode, temperature=temperature,
        center=center_thr, radius=0.12, thr_step=0.01,
        areas=(0,4,8,12,16,24)
    )
    # (선택) alpha, beta 로짓 보정 탐색
    alpha, beta, ab_f1 = tune_alpha_beta(
        model_like, val_loader, device, tta=tta_mode,
        alphas=np.linspace(0.5, 2.0, 16),
        betas=np.linspace(-1.0, 1.0, 21),
        thr=best_thr
    )
    with open(os.path.join(output_dir, "post_params.txt"), "w") as f:
        f.write(f"thr={best_thr:.3f}\n")
        f.write(f"min_area={best_min_area}\n")
        f.write(f"alpha={alpha:.3f}\n")
        f.write(f"beta={beta:.3f}\n")
    print(f"[POST] saved post_params.txt (thr={best_thr:.3f}, min_area={best_min_area}, alpha={alpha:.3f}, beta={beta:.3f})")

if __name__ == "__main__":
    main()
