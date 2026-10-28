"""Model/refinement definitions copied from ultimate.py; original file is untouched.
Only JBF behavior is corrected: equal guide/source dtype, no silent fallback.
Training, splitting and evaluation live in the other revision scripts.
"""
from typing import List, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
except ImportError:
    dcrf = None
    unary_from_softmax = None
_HAS_XIMGPROC = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "jointBilateralFilter")


@torch.no_grad()
def transductive_kmeans_refine(
    feat_map,           # (H, W, D) decoder feature map
    prob_map,           # (H, W) fused probability map (0~1)
    num_iters=5,        # number of EM / k-means iterations
    fg_thresh=0.9,      # threshold to select confident FG seeds
    bg_thresh=0.1,      # threshold to select confident BG seeds
    use_xy=True,        # whether to append (x,y) coordinates
    xy_weight=0.1,      # weight of spatial coordinates
    soft=False,         # soft or hard assignment
    temperature=0.1,   # softness for soft assignment
):
    """
    Feature-space transductive pseudo-label refinement via
    iterative k-means / EM.

    Returns:
        refined_prob_map: (H, W) refined foreground probabilities
    """

    # ============================================================
    # 1. Flatten spatial dimensions
    # ============================================================
    # Convert pixel-wise feature map into a list of feature vectors
    H, W, D = feat_map.shape
    feat = feat_map.view(-1, D)      # (N, D), N = H*W
    prob = prob_map.view(-1)         # (N,)

    device = feat.device
    N = feat.shape[0]

    # ============================================================
    # 2. Optionally append (x, y) spatial coordinates
    # ============================================================
    # This injects spatial consistency directly into feature space
    if use_xy:
        # Normalized pixel coordinates in [0, 1]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing="ij"
        )
        xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # (N, 2)

        # Concatenate spatial features to appearance features
        feat = torch.cat([feat, xy_weight * xy], dim=1)  # (N, D+2)

    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    feat = F.normalize(feat, p=2, dim=1)  # HUGE for stability

    # ============================================================
    # 3. Seed selection (VERY IMPORTANT)
    # ============================================================
    # These are your "anchors" for transductive propagation
    fg_mask = prob >= fg_thresh
    bg_mask = prob <= bg_thresh

    # Safety check: if seeds are too few, do nothing
    if fg_mask.sum() < 10 or bg_mask.sum() < 10:
        # Fallback to original probabilities
        return prob_map

    # ============================================================
    # 4. Initialize centroids (class prototypes)
    # ============================================================
    # Foreground and background feature prototypes
    fg_centroid = feat[fg_mask].mean(dim=0, keepdim=True)  # (1, D')
    bg_centroid = feat[bg_mask].mean(dim=0, keepdim=True)  # (1, D')

    # Stack as class centroids: index 0=BG, 1=FG
    centroids = torch.cat([bg_centroid, fg_centroid], dim=0)  # (2, D')
    centroids = centroids / (centroids.norm(p=2, dim=1, keepdim=True) + 1e-6)

    centroids = F.normalize(centroids, p=2, dim=1)


    # ============================================================
    # 5. EM / k-means iterations (TRANSDUCTIVE STEP)
    # ============================================================
    for _ in range(num_iters):

        # --------------------------------------------------------
        # E-step: assign each pixel to nearest centroid
        # --------------------------------------------------------
        # Pairwise distances: (N, 2)
        dist = torch.cdist(feat, centroids)

        if soft:
            # Soft assignment (probabilistic)
            assign = F.softmax(-dist / temperature, dim=1)  # (N, 2)
        else:
            # Hard assignment (k-means)
            hard_label = torch.argmin(dist, dim=1)          # (N,)
            assign = F.one_hot(hard_label, num_classes=2).float()

        # --------------------------------------------------------
        # M-step: update centroids using assigned pixels
        # --------------------------------------------------------
        for k in range(2):
            weight = assign[:, k:k+1]
            if weight.sum() > 1e-6:
                centroids[k] = (weight * feat).sum(dim=0) / weight.sum()

        # keep prototypes stable
        centroids = torch.nan_to_num(centroids, nan=0.0, posinf=0.0, neginf=0.0)
        centroids = F.normalize(centroids, p=2, dim=1)


    # ============================================================
    # 6. Extract refined foreground probability
    # ============================================================
    refined_fg_prob = assign[:, 1]       # foreground channel
    refined_prob_map = refined_fg_prob.view(H, W)

    return refined_prob_map

def _require(pkg_ok: bool, name: str, how: str):
    if not pkg_ok:
        raise RuntimeError(f"{name} is not available. Install/enable it. {how}")

def _resize_rgb_to_prob(rgb_u8: np.ndarray, H: int, W: int) -> np.ndarray:
    if rgb_u8 is None:
        return None
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got {rgb_u8.shape}")
    if (rgb_u8.shape[0] != H) or (rgb_u8.shape[1] != W):
        rgb_u8 = cv2.resize(rgb_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgb_u8, dtype=np.uint8)

def refine_prob_joint_bilateral(rgb_u8: np.ndarray,
                                prob01: np.ndarray,
                                *,
                                d: int = 7,
                                sigma_color: float = 50.0,
                                sigma_space: float = 7.0) -> np.ndarray:
    prob = np.clip(prob01, 0.0, 1.0).astype(np.float32, copy=False)
    H, W = prob.shape
    rgb_u8 = _resize_rgb_to_prob(rgb_u8, H, W)
    if _HAS_XIMGPROC:
        out = cv2.ximgproc.jointBilateralFilter(
            joint=rgb_u8.astype(np.float32),
            src=prob,
            d=int(d),
            sigmaColor=float(sigma_color),
            sigmaSpace=float(sigma_space)
        )
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    raise RuntimeError("JBF requires opencv-contrib-python-headless; refusing a different filter.")

def refine_prob_densecrf(rgb_u8: np.ndarray,
                         prob01: np.ndarray,
                         *,
                         iters: int = 5,
                         sxy_gaussian: float = 3.0,
                         compat_gaussian: float = 3.0,
                         sxy_bilateral: float = 50.0,
                         srgb_bilateral: float = 10.0,
                         compat_bilateral: float = 5.0) -> np.ndarray:
    _require(dcrf is not None and unary_from_softmax is not None,
             "pydensecrf", "Try: pip install pydensecrf")

    H, W = prob01.shape
    rgb_u8 = _resize_rgb_to_prob(rgb_u8, H, W)
    prob = np.clip(prob01, 1e-6, 1.0 - 1e-6).astype(np.float32, copy=False)

    soft = np.zeros((2, H, W), dtype=np.float32)
    soft[1] = prob
    soft[0] = 1.0 - prob

    U = unary_from_softmax(soft)
    U = np.ascontiguousarray(U)

    crf = dcrf.DenseCRF2D(W, H, 2)
    crf.setUnaryEnergy(U)
    crf.addPairwiseGaussian(sxy=(sxy_gaussian, sxy_gaussian), compat=compat_gaussian)
    crf.addPairwiseBilateral(
        sxy=(sxy_bilateral, sxy_bilateral),
        srgb=(srgb_bilateral, srgb_bilateral, srgb_bilateral),
        rgbim=np.ascontiguousarray(rgb_u8),
        compat=compat_bilateral
    )
    Q = crf.inference(int(iters))
    Q = np.array(Q, dtype=np.float32).reshape(2, H, W)
    return np.clip(Q[1], 0.0, 1.0).astype(np.float32, copy=False)

def refine_prob(rgb_u8: np.ndarray,
                prob01: np.ndarray,
                method: str,
                *,
                jbf_d: int = 7,
                jbf_sigma_color: float = 50.0,
                jbf_sigma_space: float = 7.0,
                crf_iters: int = 5,
                crf_sxy_gaussian: float = 3.0,
                crf_compat_gaussian: float = 3.0,
                crf_sxy_bilateral: float = 50.0,
                crf_srgb_bilateral: float = 10.0,
                crf_compat_bilateral: float = 5.0) -> np.ndarray:
    m = (method or "none").lower()
    if m == "none":
        return np.clip(prob01, 0.0, 1.0).astype(np.float32, copy=False)
    if m == "jbf":
        return refine_prob_joint_bilateral(rgb_u8, prob01, d=jbf_d, sigma_color=jbf_sigma_color, sigma_space=jbf_sigma_space)
    if m == "crf":
        return refine_prob_densecrf(
            rgb_u8, prob01,
            iters=crf_iters,
            sxy_gaussian=crf_sxy_gaussian,
            compat_gaussian=crf_compat_gaussian,
            sxy_bilateral=crf_sxy_bilateral,
            srgb_bilateral=crf_srgb_bilateral,
            compat_bilateral=crf_compat_bilateral
        )
    if m == "both":
        p = refine_prob_joint_bilateral(rgb_u8, prob01, d=jbf_d, sigma_color=jbf_sigma_color, sigma_space=jbf_sigma_space)
        p = refine_prob_densecrf(
            rgb_u8, p,
            iters=crf_iters,
            sxy_gaussian=crf_sxy_gaussian,
            compat_gaussian=crf_compat_gaussian,
            sxy_bilateral=crf_sxy_bilateral,
            srgb_bilateral=crf_srgb_bilateral,
            compat_bilateral=crf_compat_bilateral
        )
        return p
    raise ValueError(f"Unknown refine method: {method}")

def build_model(name, enc, in_ch=3, encoder_weights="imagenet"):
    kw = dict(encoder_name=enc, in_channels=in_ch, classes=1, encoder_weights=encoder_weights)
    name = name.lower()
    if name=="unet": return smp.Unet(**kw)
    if name=="unetplusplus": return smp.UnetPlusPlus(**kw)
    if name=="fpn": return smp.FPN(**kw)
    if name=="deeplabv3": return smp.DeepLabV3(**kw)
    if name=="deeplabv3plus": return smp.DeepLabV3Plus(**kw)
    if name=="pan": return smp.PAN(**kw)
    if name=="linknet": return smp.Linknet(**kw)
    raise ValueError(f"Unknown model {name}")

@torch.no_grad()
def extract_decoder_feat_smp(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Extract a decoder feature map from an SMP segmentation model.

    Returns:
        dec: (B, C, H, W) feature map aligned to input spatial resolution.
    """
    # 1) Encoder returns a list of multi-scale features
    feats = model.encoder(x)

    # 2) Different SMP decoders have different calling conventions:
    #    - Unet/FPN often expect decoder(*feats)
    #    - DeepLabV3(+) often expects decoder(feats)
    try:
        dec = model.decoder(*feats)
    except TypeError:
        dec = model.decoder(feats)

    # 3) Some decoders return (x, ...) tuples; keep the tensor
    if isinstance(dec, (list, tuple)):
        dec = dec[0]

    # 4) Ensure spatial size matches the network input size (H,W)
    dec = F.interpolate(dec, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return dec

class HeadBlock(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 64, dropout: float = 0.0, dil: int = 1):
        super().__init__()
        pad = dil
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=pad, dilation=dil, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(mid_ch, 1, kernel_size=1)
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadSMP(nn.Module):
    """
    Multi-head segmentation on top of the *decoder feature map* of an SMP model.
    Works for Unet/FPN/DeepLabV3+ etc. as long as .encoder and .decoder exist.
    """
    def __init__(self, arch: str, encoder_name: str, in_ch: int = 3,
                 head_cfgs=((64,0.0,1),(64,0.1,1),(96,0.0,2)),
                 encoder_weights="imagenet"):
        super().__init__()
        self.arch = arch
        self.encoder_name = encoder_name
        self.base = build_model(arch, encoder_name, in_ch=in_ch, encoder_weights=encoder_weights)

        seg_head = self.base.segmentation_head
        if isinstance(seg_head, nn.Sequential):
            dec_out_ch = seg_head[0].in_channels
        else:
            dec_out_ch = seg_head.in_channels

        self.heads = nn.ModuleList([
            HeadBlock(dec_out_ch, mid_ch=h[0], dropout=h[1], dil=h[2])
            for h in head_cfgs
        ])

    def forward(self, x):
        feats = self.base.encoder(x)

        # SMP decoder call compatibility
        dec_forward = self.base.decoder.forward
        argcount = dec_forward.__code__.co_argcount  # includes self
        if argcount == 2:
            dec = self.base.decoder(feats)
        else:
            dec = self.base.decoder(*feats)

        outs = [h(dec) for h in self.heads]  # list of [B,1,H,W]
        # Upsample head logits to match input resolution (and masks)
        H, W = x.shape[2], x.shape[3]
        outs = [F.interpolate(o, size=(H, W), mode="bilinear", align_corners=False) for o in outs]

        return outs

class SingleHeadSMP(nn.Module):
    """
    Plain SMP model (no custom multi-head).
    Returns a LIST with one element to be compatible with your training/eval code.
    """
    def __init__(self, arch: str, encoder_name: str, in_ch: int = 3, encoder_weights="imagenet"):
        super().__init__()
        self.arch = arch
        self.encoder_name = encoder_name
        self.base = build_model(arch, encoder_name, in_ch=in_ch, encoder_weights=encoder_weights)

    def forward(self, x):
        # SMP models output (B,1,H,W) logits already (since classes=1)
        logits = self.base(x)
        # Return list for compatibility with fuse_heads_logits/train_epoch_base/eval_metrics_base
        return [logits]

def fuse_heads_logits(logits_list: List[torch.Tensor]) -> torch.Tensor:
    probs_each = [torch.sigmoid(l).clamp(1e-6,1-1e-6) for l in logits_list]
    logits_each = [torch.logit(p) for p in probs_each]
    L = torch.stack(logits_each, 0).mean(0)
    return L  # fused logit

def pseudo_confidence(prob: np.ndarray, tau: float,
                      *,
                      min_area: float = 0.002,
                      max_area: float = 0.6,
                      ent_w: float = 0.3) -> Tuple[float, float]:
    """
    Return (score, area). If rejected -> (negative score, area).
    Same spirit as your older script:
      - reject tiny/huge predicted fg area
      - score = mean(fg prob) - ent_w * entropy(fg prob)
    """
    prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32, copy=False)

    m = (prob >= float(tau))
    area = float(m.mean())

    if (area < float(min_area)) or (area > float(max_area)) or (not m.any()):
        return -1.0, area

    fg = prob[m]
    # entropy on foreground probabilities
    ent = float(-(fg*np.log(fg + 1e-6) + (1.0 - fg)*np.log(1.0 - fg + 1e-6)).mean())
    score = float(fg.mean() - float(ent_w) * ent)
    return score, area
