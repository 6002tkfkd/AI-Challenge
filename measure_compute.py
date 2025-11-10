# measure_compute.py
import argparse, time, torch
from ptflops import get_model_complexity_info

# 네 프로젝트 경로/모델에 맞게 import
from models.hrsegnet_b16_cbam import HrSegNetB16CBAM as Net

def measure_ptflops(model, input_c=1, H=192, W=192, device="cuda"):
    model.eval().to(device)
    macs, params = get_model_complexity_info(
        model, (input_c, H, W),
        as_strings=False, print_per_layer_stat=False, verbose=False
    )
    # ptflops는 MACs(=Multiply-Accumulate) 기준을 GMac으로 주는 편
    return macs, params  # 둘 다 float (연산량=MACs, 파라미터 수=개수)

@torch.no_grad()
def measure_latency(model, input_c=1, H=192, W=192, device="cuda", runs=100, warmup=20):
    model.eval().to(device)
    x = torch.randn(1, input_c, H, W, device=device)
    # warmup
    for _ in range(warmup):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / runs * 1000.0
    return ms

def humanize(n):
    # 숫자를 보기 좋게
    if n >= 1e12: return f"{n/1e12:.2f} TM"
    if n >= 1e9:  return f"{n/1e9:.2f} GM"
    if n >= 1e6:  return f"{n/1e6:.2f} MM"
    if n >= 1e3:  return f"{n/1e3:.2f} k"
    return f"{n:.0f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=192)
    ap.add_argument("--W", type=int, default=192)
    ap.add_argument("--in-ch", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    model = Net(in_channels=args.in_ch, base=30, num_classes=1)

    macs, params = measure_ptflops(model, args.in_ch, args.H, args.W, args.device)
    # macs 단위: 연산 횟수(곱-누산). ptflops는 보통 "MACs"로 표기 (곱+덧셈 1회로 계산).
    # 흔히 FLOPs를 쓸 때는 1 MAC ≈ 2 FLOPs 로 보는 관행이 있음.

    ms = measure_latency(model, args.in_ch, args.H, args.W, args.device, args.runs)

    print("=== Net @ {}x{} (C={}) ===".format(args.H, args.W, args.in_ch))
    print(f"Params: {humanize(params)} params  (~{params/1e6:.2f} M)")
    print(f"MACs:   {humanize(macs)} MACs     (~{macs/1e6:.1f} MMAC)")
    print(f"Latency: {ms:.2f} ms  (batch=1)")

if __name__ == "__main__":
    main()
