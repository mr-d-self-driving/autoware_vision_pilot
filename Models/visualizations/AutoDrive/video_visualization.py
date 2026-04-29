from __future__ import annotations

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

from Models.data_utils.load_data_auto_drive import CURV_SCALE  # noqa: E402
from Models.model_components.autodrive.autodrive_network import AutoDrive  # noqa: E402

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_NET_W, _NET_H = 1024, 512
_D_MAX_M = 150.0


def _preprocess(frame_bgr) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((_NET_W, _NET_H), Image.BILINEAR)
    t = TF.to_tensor(pil)
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


def _draw_overlay(
    frame_bgr,
    frame_idx: int,
    pred_dist_m: float,
    pred_curv_1pm: float,
    pred_steer_deg: float,
    pred_flag_prob: float,
    pred_flag_cls: int,
):
    out = frame_bgr.copy()
    cv2.rectangle(out, (18, 18), (860, 235), (20, 20, 20), -1)
    cv2.rectangle(out, (18, 18), (860, 235), (120, 120, 120), 2)
    lines = [
        f"AutoDrive  |  Frame {frame_idx}",
        f"Distance (m): {pred_dist_m:.2f}",
        f"Curvature (1/m): {pred_curv_1pm:+.5f}",
        f"Steering (deg): {pred_steer_deg:+.2f}",
        f"CIPO Flag: {pred_flag_cls}  (prob={pred_flag_prob:.3f})",
    ]
    y0 = 55
    for i, txt in enumerate(lines):
        color = (240, 240, 240) if i == 0 else (255, 255, 255)
        cv2.putText(out, txt, (38, y0 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
    return out


def main():
    p = ArgumentParser(description="AutoDrive video visualization")
    p.add_argument("--checkpoint", required=True, help="AutoDrive .pth checkpoint")
    p.add_argument("--video",      required=True, help="Input video path")
    p.add_argument("--output",     required=True, help="Output video path (.avi/.mp4)")
    p.add_argument("--start",      type=int,   default=0,    help="Start frame index")
    p.add_argument("--max-frames", type=int,   default=0,    help="Max frames to process (0 = all)")
    p.add_argument("--fps",        type=float, default=20.0, help="Output FPS")
    p.add_argument("--flag-threshold", type=float, default=0.65, help="CIPO flag threshold")
    p.add_argument("--wheelbase-m",    type=float, default=2.984, help="Wheelbase (m)")
    p.add_argument("--steer-ratio",    type=float, default=16.8,  help="Steering column ratio")
    p.add_argument("--vis", action="store_true", default=False, help="Show live preview window")
    args = p.parse_args()

    cap = cv2.VideoCapture(str(Path(args.video).expanduser().resolve()))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    in_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    fps   = args.fps if args.fps > 0 else (in_fps if in_fps > 0 else 20.0)

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (_NET_W, _NET_H)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {out_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _load_model(Path(args.checkpoint).expanduser().resolve(), device)

    ok, prev = cap.read()
    if not ok:
        cap.release(); writer.release()
        raise RuntimeError("No frames in input video")

    frame_idx = 0
    processed = 0
    print(f"Processing video ({in_w}x{in_h}) on {device} ...")

    while True:
        ok, curr = cap.read()
        if not ok:
            break
        frame_idx += 1

        if frame_idx < args.start:
            prev = curr
            continue
        if args.max_frames > 0 and processed >= args.max_frames:
            break

        t_prev = _preprocess(prev).unsqueeze(0).to(device)
        t_curr = _preprocess(curr).unsqueeze(0).to(device)
        with torch.no_grad():
            d_pred, curv_pred, flag_logit = model(t_prev, t_curr)

        pred_d_norm    = float(d_pred.squeeze().cpu())
        pred_dist_m    = _D_MAX_M * (1.0 - max(0.0, min(1.0, pred_d_norm)))
        pred_curv_1pm  = float(curv_pred.squeeze().cpu()) * CURV_SCALE
        pred_flag_prob = float(torch.sigmoid(flag_logit.squeeze()).cpu())
        pred_flag_cls  = 1 if pred_flag_prob >= args.flag_threshold else 0
        pred_steer_deg = _curvature_to_steer_deg(pred_curv_1pm, args.wheelbase_m, args.steer_ratio)

        vis = cv2.resize(curr, (_NET_W, _NET_H), interpolation=cv2.INTER_LINEAR)
        vis = _draw_overlay(vis, frame_idx, pred_dist_m, pred_curv_1pm,
                            pred_steer_deg, pred_flag_prob, pred_flag_cls)
        writer.write(vis)

        if args.vis:
            cv2.imshow("AutoDrive", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        processed += 1
        prev = curr

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done — {processed} frames saved to: {out_path}")


if __name__ == "__main__":
    main()
