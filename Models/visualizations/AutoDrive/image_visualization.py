from __future__ import annotations

import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from Models.data_utils.load_data_auto_drive import CURV_SCALE  # noqa: E402
from Models.model_components.autodrive.autodrive_network import AutoDrive  # noqa: E402

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_NET_W, _NET_H = 1024, 512
_D_MAX_M = 150.0


def _preprocess_for_autodrive(img: Image.Image) -> torch.Tensor:
    img = img.resize((_NET_W, _NET_H), Image.BILINEAR)
    t = TF.to_tensor(img)
    return TF.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD)


def _curvature_to_steer_deg(curv_1pm: float, wheelbase_m: float, steer_ratio: float) -> float:
    return math.degrees(math.atan(curv_1pm * wheelbase_m) * steer_ratio)


def _load_model(checkpoint: Path, device: torch.device) -> AutoDrive:
    model = AutoDrive().to(device)
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _overlay_info(
    frame_bgr,
    pred_dist_m: float,
    pred_curv_1pm: float,
    pred_steer_deg: float,
    pred_flag_prob: float,
    pred_flag_cls: int,
):
    out = frame_bgr.copy()
    cv2.rectangle(out, (20, 20), (780, 230), (20, 20, 20), -1)
    cv2.rectangle(out, (20, 20), (780, 230), (110, 110, 110), 2)
    fs = 0.82
    lh = 36
    x = 40
    y0 = 58
    lines = [
        "AutoDrive Inference (Generic Input)",
        f"Distance (m): {pred_dist_m:.2f}",
        f"Curvature (1/m): {pred_curv_1pm:+.5f}",
        f"Steering (deg): {pred_steer_deg:+.2f}",
        f"CIPO Flag: {pred_flag_cls}  (prob={pred_flag_prob:.3f})",
    ]
    for i, txt in enumerate(lines):
        color = (230, 230, 230) if i == 0 else (255, 255, 255)
        cv2.putText(out, txt, (x, y0 + i * lh), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2, cv2.LINE_AA)
    return out


def main():
    p = ArgumentParser(description="AutoDrive image visualization (generic unseen data)")
    p.add_argument("--checkpoint", required=True, help="AutoDrive .pth checkpoint")
    p.add_argument("--image-prev", required=True, help="Previous image filepath")
    p.add_argument("--image-curr", required=True, help="Current image filepath")
    p.add_argument("--flag-threshold", type=float, default=0.65, help="Flag threshold")
    p.add_argument("--wheelbase-m", type=float, default=2.984, help="Wheelbase (m)")
    p.add_argument("--steer-ratio", type=float, default=16.8, help="Steering column ratio")
    p.add_argument("--save", default="", help="Optional output image path")
    args = p.parse_args()

    prev_path = Path(args.image_prev).expanduser().resolve()
    curr_path = Path(args.image_curr).expanduser().resolve()
    if not prev_path.exists() or not curr_path.exists():
        raise FileNotFoundError("image-prev or image-curr path does not exist")

    img_prev = Image.open(prev_path).convert("RGB")
    img_curr = Image.open(curr_path).convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(Path(args.checkpoint).expanduser().resolve(), device)

    t_prev = _preprocess_for_autodrive(img_prev).unsqueeze(0).to(device)
    t_curr = _preprocess_for_autodrive(img_curr).unsqueeze(0).to(device)
    with torch.no_grad():
        d_pred, curv_pred, flag_logit = model(t_prev, t_curr)

    pred_d_norm = float(d_pred.squeeze().cpu())
    pred_dist_m = _D_MAX_M * (1.0 - max(0.0, min(1.0, pred_d_norm)))
    pred_curv_1pm = float(curv_pred.squeeze().cpu()) * CURV_SCALE
    pred_flag_prob = float(torch.sigmoid(flag_logit.squeeze()).cpu())
    pred_flag_cls = 1 if pred_flag_prob >= args.flag_threshold else 0
    pred_steer_deg = _curvature_to_steer_deg(pred_curv_1pm, args.wheelbase_m, args.steer_ratio)

    vis_rgb = np.array(img_curr.resize((_NET_W, _NET_H), Image.BILINEAR))
    vis = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
    vis = _overlay_info(vis, pred_dist_m, pred_curv_1pm, pred_steer_deg, pred_flag_prob, pred_flag_cls)

    if args.save:
        out_path = Path(args.save).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)
        print(f"Saved visualization to: {out_path}")

    cv2.imshow("AutoDrive Image Visualization", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from Models.data_utils.load_data_auto_drive import (  # noqa: E402
    _center_crop_50deg_resize,
    _read_hfov_deg,
    CURV_SCALE,
)
from Models.model_components.autodrive.autodrive_network import AutoDrive  # noqa: E402

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_D_MAX_M = 150.0


def _to_tensor(img_np):
    t = TF.to_tensor(img_np)
    return TF.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD)


def _curvature_to_steer_deg(curv_1pm: float, wheelbase_m: float, steer_ratio: float) -> float:
    return math.degrees(math.atan(curv_1pm * wheelbase_m) * steer_ratio)


def _load_model(checkpoint: Path, device: torch.device) -> AutoDrive:
    model = AutoDrive().to(device)
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _overlay_info(
    frame_bgr,
    pred_dist_m: float,
    pred_curv_1pm: float,
    pred_steer_deg: float,
    pred_flag_prob: float,
    pred_flag_cls: int,
):
    out = frame_bgr.copy()
    cv2.rectangle(out, (20, 20), (760, 230), (20, 20, 20), -1)
    cv2.rectangle(out, (20, 20), (760, 230), (110, 110, 110), 2)
    fs = 0.8
    lh = 36
    x = 40
    y0 = 58
    lines = [
        "AutoDrive Inference",
        f"Distance (m): {pred_dist_m:.2f}",
        f"Curvature (1/m): {pred_curv_1pm:+.5f}",
        f"Steering (deg): {pred_steer_deg:+.2f}",
        f"CIPO Flag: {pred_flag_cls}  (prob={pred_flag_prob:.3f})",
    ]
    for i, txt in enumerate(lines):
        color = (230, 230, 230) if i == 0 else (255, 255, 255)
        cv2.putText(out, txt, (x, y0 + i * lh), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2, cv2.LINE_AA)
    return out


def _load_label_from_image(zod_root: Path, seq: str, image_name: str) -> dict | None:
    stem = Path(image_name).stem
    label_path = zod_root / "labels" / seq / f"{stem}.json"
    if not label_path.exists():
        return None
    with open(label_path, "r") as f:
        return json.load(f)


def main():
    p = ArgumentParser(description="AutoDrive image visualization")
    p.add_argument("--root", required=True, help="ZOD root")
    p.add_argument("--checkpoint", required=True, help="AutoDrive .pth checkpoint")
    p.add_argument("--sequence", required=True, help="Sequence id, e.g. 001451")
    p.add_argument("--image-prev", required=True, help="Prev frame image filename")
    p.add_argument("--image-curr", required=True, help="Curr frame image filename")
    p.add_argument("--flag-threshold", type=float, default=0.65, help="Flag threshold")
    p.add_argument("--wheelbase-m", type=float, default=2.984, help="Wheelbase (m)")
    p.add_argument("--steer-ratio", type=float, default=16.8, help="Steering column ratio")
    p.add_argument("--save", default="", help="Optional output image path")
    args = p.parse_args()

    zod_root = Path(args.root).expanduser().resolve()
    seq = args.sequence
    img_dir = zod_root / "images_blur" / "sequences" / seq / "camera_front_blur"
    if not img_dir.exists():
        img_dir = zod_root / "images_blur_1" / "sequences" / seq / "camera_front_blur"
    if not img_dir.exists():
        raise FileNotFoundError(f"Could not find image directory for sequence {seq}")

    prev_path = img_dir / args.image_prev
    curr_path = img_dir / args.image_curr
    if not prev_path.exists() or not curr_path.exists():
        raise FileNotFoundError("image-prev or image-curr does not exist in sequence image folder")

    hfov = _read_hfov_deg(zod_root, seq)
    np_prev = _center_crop_50deg_resize(Image.open(prev_path).convert("RGB"), hfov)
    np_curr = _center_crop_50deg_resize(Image.open(curr_path).convert("RGB"), hfov)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(Path(args.checkpoint).expanduser().resolve(), device)

    t_prev = _to_tensor(np_prev).unsqueeze(0).to(device)
    t_curr = _to_tensor(np_curr).unsqueeze(0).to(device)
    with torch.no_grad():
        d_pred, curv_pred, flag_logit = model(t_prev, t_curr)

    pred_d_norm = float(d_pred.squeeze().cpu())
    pred_dist_m = _D_MAX_M * (1.0 - max(0.0, min(1.0, pred_d_norm)))
    pred_curv_1pm = float(curv_pred.squeeze().cpu()) * CURV_SCALE
    pred_flag_prob = float(torch.sigmoid(flag_logit.squeeze()).cpu())
    pred_flag_cls = 1 if pred_flag_prob >= args.flag_threshold else 0
    pred_steer_deg = _curvature_to_steer_deg(pred_curv_1pm, args.wheelbase_m, args.steer_ratio)

    vis = cv2.cvtColor(np_curr, cv2.COLOR_RGB2BGR)
    vis = _overlay_info(
        vis, pred_dist_m, pred_curv_1pm, pred_steer_deg, pred_flag_prob, pred_flag_cls
    )

    lbl = _load_label_from_image(zod_root, seq, args.image_curr)
    if lbl is not None:
        gt_flag = 1 if lbl.get("cipo_detected") else 0
        gt_dist = lbl.get("distance_to_in_path_object")
        gt_curv = float(lbl.get("curvature") or 0.0)
        gt_steer = _curvature_to_steer_deg(gt_curv, args.wheelbase_m, args.steer_ratio)
        y = 270
        cv2.rectangle(vis, (20, y), (760, y + 145), (20, 20, 20), -1)
        cv2.rectangle(vis, (20, y), (760, y + 145), (110, 110, 110), 2)
        gt_lines = [
            "Ground Truth",
            f"GT CIPO Flag: {gt_flag}",
            f"GT Distance (m): {float(gt_dist):.2f}" if gt_dist is not None else "GT Distance (m): N/A",
            f"GT Curvature (1/m): {gt_curv:+.5f}",
            f"GT Steering (deg): {gt_steer:+.2f}",
        ]
        for i, txt in enumerate(gt_lines):
            cv2.putText(vis, txt, (40, y + 36 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if args.save:
        out_path = Path(args.save).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)
        print(f"Saved visualization to: {out_path}")

    cv2.imshow("AutoDrive Image Visualization", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
