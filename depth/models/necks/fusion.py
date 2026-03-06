# necks/hf_contour_fusion_neck.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_

from mmcv.runner import BaseModule
from mmcv.ops import ModulatedDeformConv2d
from depth.models.builder import NECKS


class HFAlign(BaseModule):
    def __init__(self, out_ch, hf_channels):
        super().__init__()
        self.proj = nn.Conv2d(hf_channels, out_ch, 1, bias=False)

    def forward(self, hf, size_hw):
        a =hf
        x = F.interpolate(hf, size=size_hw, mode='bilinear', align_corners=False)
        return self.proj(x)

class EdgeGate(BaseModule):
    def __init__(self, c, hf_channels=192):
        super().__init__()
        self.hf_align = HFAlign(c, hf_channels)
        self.gate = nn.Sequential(nn.Conv2d(c, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, hf, feat):  # feat: [B,C,h,w]
        h, w = feat.shape[-2:]
        hfi = self.hf_align(hf, (h, w))
        m = self.gate(hfi)
        return feat * (1.0 + m)
    
class EdgeInjector(BaseModule):
    def __init__(self, c, hf_channels=192):
        super().__init__()
        self.hf_align = HFAlign(c, hf_channels)
        
        self.hf_transform = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False), # DW卷积，极度轻量且保持边缘
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=1, bias=False) 
        )

        self.gate_gen = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, hf, feat):  # feat: [B,C,h,w]
        h, w = feat.shape[-2:]
        hfi = self.hf_align(hf, (h, w)) # [B, C, H, W]
        hf_feature = self.hf_transform(hfi)
        mask = self.gate_gen(hfi)
        return feat + self.scale * (hf_feature * mask)
    
class ConcatFuse(BaseModule):
    def __init__(self, c, hf_channels=192):
        super().__init__()
        self.hf_align = HFAlign(c, hf_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(c + c, c, 1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(c)
        )

    def forward(self, hf, feat):
        h, w = feat.shape[-2:]
        hfi = self.hf_align(hf, (h, w))
        return self.fuse(torch.cat([feat, hfi], dim=1))

class FiLMFuse(BaseModule):
    def __init__(self, c, hf_channels=192, spatial=False):
        super().__init__()
        self.hf_align = HFAlign(c, hf_channels)
        self.spatial = spatial
        if spatial:
            self.affine = nn.Conv2d(c, 2*c, 3, padding=1)
        else:
            self.affine = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, 2*c, 1))

    def forward(self, hf, feat):
        h, w = feat.shape[-2:]
        hfi = self.hf_align(hf, (h, w))
        ab = self.affine(hfi)
        gamma, beta = torch.chunk(ab, 2, dim=1)
        return gamma * feat + beta

class HFDeformFuse(BaseModule):
    def __init__(self, c, hf_channels=192, k=3):
        super().__init__()
        self.hf_align = HFAlign(c, hf_channels)
        self.offset = nn.Conv2d(c, 2*k*k, 3, padding=1)
        self.mask   = nn.Conv2d(c,   k*k, 3, padding=1)
        self.dcn    = ModulatedDeformConv2d(c, c, k, padding=k//2, bias=False)
        self.bnact  = nn.Sequential(nn.BatchNorm2d(c), nn.GELU())

    def forward(self, hf, feat):
        h, w = feat.shape[-2:]
        hfi = self.hf_align(hf, (h, w))
        offset = self.offset(hfi)
        mask   = torch.sigmoid(self.mask(hfi))
        y = self.dcn(feat, offset, mask)
        return self.bnact(y + feat)


class HFWinXAttn(BaseModule):
    def __init__(self, c, hf_channels=192, window_size=7, heads=None, cross_ratio=0.5):
        """
        Args:
            c (int):  Number of input feature channels.
            hf_channels (int): Number of high-frequency feature channels.
            window_size (int): Window size.
            heads (int, optional): Number of attention heads. If None, automatically matched based on c.
            cross_ratio (float): Channel ratio participating in cross-attention.
        """
        super().__init__()
        self.ws = window_size
        self.hf_align = HFAlign(c, hf_channels)

        HEADS_MAP = {192: 6, 384: 12, 768: 24, 1536: 48}
        self.c_cross_actual = int(c *cross_ratio)

        if heads is None:
            self.heads = HEADS_MAP.get(c)
            if self.heads is None:
                raise ValueError(f"in_channels c={c} not in HEADS_MAP 中: {list(HEADS_MAP.keys())}")
        else:
            self.heads = heads
        
    
        self.c_cross = int(c * cross_ratio)
        self.c_self  = c - self.c_cross
        
        self.n_head_cross = max(1, int(self.heads * cross_ratio)) if cross_ratio > 0 else 0
        self.n_head_self  = self.heads - self.n_head_cross

        if self.n_head_self <= 0 and self.n_head_cross > 0:
            self.n_head_cross = self.heads
            self.n_head_self = 0
            self.c_cross = c
            self.c_self = 0
        
        self.norm_feat = nn.LayerNorm(c)
        self.norm_hf   = nn.LayerNorm(c)

        if self.c_cross > 0 and self.n_head_cross > 0:
            self.attn_cross = nn.MultiheadAttention(self.c_cross, self.n_head_cross)
        
        if self.c_self > 0 and self.n_head_self > 0:
            self.attn_self = nn.MultiheadAttention(self.c_self, self.n_head_self)

        self.proj = nn.Linear(c, c)
        self.channel_score_feat = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(c // 4, c, 1, bias=False)
        )
        self.channel_score_hf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(c // 4, c, 1, bias=False)
        )
        self.hf_cross_align = nn.Conv2d(c, self.c_cross_actual, kernel_size=1, bias=False)
        self.local_detail = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c, 1, bias=False)
        )
        self.channel_score_joint = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # B, 2C, 1, 1
            nn.Conv2d(2*c, c//2, 1),
            nn.GELU(),
            nn.Conv2d(c//2, c, 1),
        )
        self.gate_fc = nn.Sequential(
            nn.Linear(2 * c, c // 4),
            nn.GELU(),
            nn.Linear(c // 4, c),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.proj.weight, 0)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)
        if hasattr(self.gate_fc, '2') and hasattr(self.gate_fc[2], 'weight'):
             nn.init.normal_(self.gate_fc[2].weight, std=0.001)

    # ... (其他方法 _pad_to_window, _win_part, _win_reverse, forward 保持不变) ...
    def _pad_to_window(self, x, ws):
        B, C, H, W = x.shape
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b != 0 or pad_r != 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        return x, pad_b, pad_r

    def _win_part(self, x, ws):
        x, pad_b, pad_r = self._pad_to_window(x, ws)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)
        return x, B, H, W, pad_b, pad_r

    def _win_reverse(self, x, ws, B, H, W, pad_b, pad_r):
        C = x.shape[-1]
        x = x.view(B, H // ws, W // ws, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
        if pad_b > 0: x = x[:, :, :-pad_b, :]
        if pad_r > 0: x = x[:, :, :, :-pad_r]
        return x

    def forward(self, hf, feat):
        """
        hf   : [B, C, H, W]
        feat : [B, C, H, W]
        """
        B, C, H, W = feat.shape
        C_half = C // 2   

        hf = self.hf_align(hf, (H, W))

        feat_n = self.norm_feat(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        hf_n   = self.norm_hf(hf.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        idx_expand = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        feat_sorted = torch.gather(feat_n, dim=1, index=idx_expand)
        hf_sorted   = torch.gather(hf_n,   dim=1, index=idx_expand)
      
        feat_cross = feat_sorted[:, :C_half, :, :]   

        # hf_n   = self.norm_hf(hf.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat_self  = feat_sorted[:, C_half:, :, :]  

        hf_cross   = hf_sorted[:, :C_half, :, :]

        x_cross, Bq, Hq, Wq, pad_b, pad_r = self._win_part(feat_cross, self.ws)
        x_self,  _,  _,  _,  _,     _     = self._win_part(feat_self,  self.ws)
        hf_cross, _, _, _, _, _          = self._win_part(hf_cross,   self.ws)

        q_cross = x_cross * (1.0 / 0.07)

        out_cross, _ = self.attn_cross(
            q_cross,
            hf_cross,
            hf_cross,
            need_weights=False
        )

        out_self, _ = self.attn_self(
            x_self,
            x_self,
            x_self,
            need_weights=False
        )

        attn_out = torch.cat([out_cross, out_self], dim=-1)
        attn_out = self.proj(attn_out)
        y_attn = self._win_reverse(attn_out, self.ws, Bq, Hq, Wq, pad_b, pad_r)
        y_detail = self.local_detail(hf_n)
        gate_in = torch.cat([feat_n, y_attn], dim=1).permute(0, 2, 3, 1)
        gate = self.gate_fc(gate_in).permute(0, 3, 1, 2)

        out = feat + gate * y_attn + y_detail

        return out
    
class AdaptiveDownsampleFusion(nn.Module):

    def __init__(self, in_ch_prev, in_ch_curr):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch_prev, in_ch_curr, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_ch_curr),
            nn.GELU()
        )
        

        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_ch_curr * 2, in_ch_curr, 1, bias=False),
            nn.BatchNorm2d(in_ch_curr),
            nn.GELU(),
            nn.Conv2d(in_ch_curr, 1, 3, padding=1, bias=True), # Spatial attention
            nn.Sigmoid()
        )
        self.proj =nn.Sequential(nn.Conv2d(in_ch_prev,in_ch_curr,kernel_size=1),
                                 nn.GELU())

    def forward(self, prev_feat, curr_feat):
        prev_feat = self.downsample(prev_feat) 
        cat_feat = torch.cat([curr_feat, prev_feat], dim=1)
        mask = self.gate_conv(cat_feat) # [B, 1, H, W]

        return curr_feat + mask * prev_feat

@NECKS.register_module()
class HFContourFusionNeck(BaseModule):
    """outs = [hf, f1, f2, f3, f4] 或 [stem64, hf, f1, f2, f3, f4]"""
    def __init__(self,
                 in_channels=(192,384,768,1536),
                 hf_channels=192,
                 methods=('winxattn','winxattn','winxattn','winxattn'),
                 window_size=7,
                 heads=(6,12,24,48),
                 with_stem=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        assert len(in_channels)==4 and len(methods)==4
        self.hf_channels = hf_channels
        self.with_stem = with_stem
        self.fusers = nn.ModuleList()
        for c, m, h in zip(in_channels, methods,heads):
            if m == 'edge':
                mod = EdgeInjector(c, hf_channels)
            elif m == 'concat':
                mod = ConcatFuse(c, hf_channels)
            elif m == 'film':
                mod = FiLMFuse(c, hf_channels, spatial=False)
            elif m == 'winxattn':
                mod = HFWinXAttn(c, hf_channels, window_size, h)
            elif m == 'deform':
                mod = HFDeformFuse(c, hf_channels, k=3)
            else:
                raise ValueError(f'Unknown method: {m}')
            self.fusers.append(mod)
        self.cascade_links = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            link = AdaptiveDownsampleFusion(in_channels[i], in_channels[i+1])
            self.cascade_links.append(link)

    def init_weights(self):
        super().init_weights()
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    xavier_uniform_(m.weight)
                    if getattr(m, 'bias', None) is not None and m.bias is not None:
                        constant_(m.bias, 0.)
                elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.LayerNorm)):
                    constant_(m.weight, 1.0)
                    constant_(m.bias, 0.0)

    def forward(self, outs, return_feats=False):
        hf, f1, f2, f3, f4 = outs
        feats = [f1, f2, f3, f4]
        fused = []
        last_out = None

        for i in range(len(feats)):
            curr_feat = feats[i]
            if i > 0 and last_out is not None:
                curr_feat = self.cascade_links[i-1](last_out, curr_feat)
            out = self.fusers[i](hf,curr_feat)
            fused.append(out)
            last_out = out
 
        return [*fused]

