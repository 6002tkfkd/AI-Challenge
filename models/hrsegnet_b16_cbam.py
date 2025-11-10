# models/hrsegnet_b16_cbam_gn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# GroupNorm helper
# ---------------------------
def _pick_num_groups(c: int) -> int:
    # pick largest divisor from this set
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1

def GN(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=_pick_num_groups(c), num_channels=c)

# ---------------------------
# CBAM (Channel + Spatial)
# ---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # 전역 평균/최댓값 풀링을 연산으로 대체 (결정론)
        avg = x.mean(dim=(2, 3), keepdim=True)           # H,W 평균
        mx  = torch.amax(x, dim=(2, 3), keepdim=True)    # H,W 최댓값
        out = self.mlp(avg) + self.mlp(mx)
        return self.sig(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg, mx], dim=1)
        attn = self.sig(self.conv(x_cat))
        return attn

class CBAM(nn.Module):
    def __init__(self, in_planes, ca_ratio=8, sa_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio=ca_ratio)
        self.sa = SpatialAttention(kernel_size=sa_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ---------------------------
# SegHead (원본과 동일 인터페이스, BN→GN)
# ---------------------------
class SegHead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, aux_head=False):
        super().__init__()
        self.bn1 = GN(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if aux_head:
            self.con_bn_relu = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=interplanes, kernel_size=3, stride=1, padding=1),
                GN(interplanes),
                nn.ReLU(inplace=True),
            )
        else:
            self.con_bn_relu = nn.Sequential(
                nn.ConvTranspose2d(in_channels=inplanes, out_channels=interplanes,
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                GN(interplanes),
                nn.ReLU(inplace=True),
            )
        self.conv = nn.Conv2d(in_channels=interplanes, out_channels=outplanes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.con_bn_relu(x)
        out = self.conv(x)
        return out

# ---------------------------
# HrSegNetB16의 각 stage 블록 + CBAM
# (BN→GN 치환)
# ---------------------------
class SegBlockCBAM(nn.Module):
    def __init__(self, base=30, stage_index=1, cbam_ratio=8, cbam_kernel=7):
        super().__init__()
        # high-res convs
        self.h_conv1 = nn.Sequential(nn.Conv2d(base, base, 3, 1, 1), GN(base), nn.ReLU(inplace=True))
        self.h_conv2 = nn.Sequential(nn.Conv2d(base, base, 3, 1, 1), GN(base), nn.ReLU(inplace=True))
        self.h_conv3 = nn.Sequential(nn.Conv2d(base, base, 3, 1, 1), GN(base), nn.ReLU(inplace=True))

        # low-res branch per stage
        outc = base * int(math.pow(2, stage_index))
        if stage_index == 1:
            self.l_conv1 = nn.Sequential(
                nn.Conv2d(base, outc, 3, 2, 1),
                GN(outc),
                nn.ReLU(inplace=True)
            )
        elif stage_index == 2:
            self.l_conv1 = nn.Sequential(
                nn.AvgPool2d(3, 2, 1),
                nn.Conv2d(base, outc, 3, 2, 1),
                GN(outc),
                nn.ReLU(inplace=True)
            )
        elif stage_index == 3:
            self.l_conv1 = nn.Sequential(
                nn.AvgPool2d(3, 2, 1),
                nn.Conv2d(base, outc, 3, 2, 1),
                GN(outc),
                nn.ReLU(inplace=True),
                nn.Conv2d(outc, outc, 3, 2, 1),
                GN(outc),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError("stage_index must be 1, 2, or 3")

        self.l_conv2 = nn.Sequential(nn.Conv2d(outc, outc, 3, 1, 1), GN(outc), nn.ReLU(inplace=True))
        self.l_conv3 = nn.Sequential(nn.Conv2d(outc, outc, 3, 1, 1), GN(outc), nn.ReLU(inplace=True))

        # adapters low->high
        self.l2h_conv1 = nn.Conv2d(outc, base, 1, 1, 0)
        self.l2h_conv2 = nn.Conv2d(outc, base, 1, 1, 0)
        self.l2h_conv3 = nn.Conv2d(outc, base, 1, 1, 0)

        # stage 출력에 CBAM 1회
        self.cbam = CBAM(in_planes=base, ca_ratio=cbam_ratio, sa_kernel=cbam_kernel)

    def forward(self, x):
        size = x.shape[2:]

        out_h1 = self.h_conv1(x)
        out_l1 = self.l_conv1(x)
        out_l1_up = F.interpolate(out_l1, size=size, mode='bilinear', align_corners=True)
        out_hl1 = self.l2h_conv1(out_l1_up) + out_h1

        out_h2 = self.h_conv2(out_hl1)
        out_l2 = self.l_conv2(out_l1)
        out_l2_up = F.interpolate(out_l2, size=size, mode='bilinear', align_corners=True)
        out_hl2 = self.l2h_conv2(out_l2_up) + out_h2

        out_h3 = self.h_conv3(out_hl2)
        out_l3 = self.l_conv3(out_l2)
        out_l3_up = F.interpolate(out_l3, size=size, mode='bilinear', align_corners=True)
        out_hl3 = self.l2h_conv3(out_l3_up) + out_h3

        # CBAM
        out = self.cbam(out_hl3)
        return out

# ---------------------------
# HrSegNetB16 + CBAM (GN 버전)
# ---------------------------
class HrSegNetB16CBAM(nn.Module):
    def __init__(self, in_channels=1, base=30, num_classes=1, cbam_ratio=8, cbam_kernel=7):
        super().__init__()
        self.base = base
        # stem (BN→GN)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base // 2, 3, 2, 1),
            GN(base // 2),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(base // 2, base, 3, 2, 1),
            GN(base),
            nn.ReLU(inplace=True),
        )

        # body: 3 stages + 각 stage 출력 CBAM
        self.seg1 = SegBlockCBAM(base=base, stage_index=1, cbam_ratio=cbam_ratio, cbam_kernel=cbam_kernel)
        self.seg2 = SegBlockCBAM(base=base, stage_index=2, cbam_ratio=cbam_ratio, cbam_kernel=cbam_kernel)
        self.seg3 = SegBlockCBAM(base=base, stage_index=3, cbam_ratio=cbam_ratio, cbam_kernel=cbam_kernel)

        # heads (BN→GN)
        self.aux_head1 = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=True)
        self.aux_head2 = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=True)
        self.head      = SegHead(inplanes=base, interplanes=base, outplanes=num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        B, C, H, W = x.shape
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        h1 = self.seg1(s2)  # CBAM 후
        h2 = self.seg2(h1)  # CBAM 후
        h3 = self.seg3(h2)  # CBAM 후

        last = self.head(h3)
        out  = F.interpolate(last, size=(H, W), mode="bilinear", align_corners=True)

        if self.training:
            a1 = self.aux_head1(h1)
            a2 = self.aux_head2(h2)
            a1 = F.interpolate(a1, size=(H, W), mode="bilinear", align_corners=True)
            a2 = F.interpolate(a2, size=(H, W), mode="bilinear", align_corners=True)
            return [out, a1, a2]
        else:
            return [out]
