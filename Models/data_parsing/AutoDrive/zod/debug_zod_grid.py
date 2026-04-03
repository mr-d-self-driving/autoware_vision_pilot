#!/usr/bin/env python3
"""
Single 2×2 debug canvas per frame: raw radar BEV | labels viz | CIPO viz | legend.

Outputs under {zod_root}/output/{seq}/debug/images/ (one PNG per sampled frame).
Replaces separate debug_labels_viz and debug_cipo_radar_viz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ZOD_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_ZOD_SCRIPT_DIR))

try:
    from Models.inference.auto_speed_infer import AutoSpeedNetworkInfer
except ImportError:
    from inference.auto_speed_infer import AutoSpeedNetworkInfer

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

_LAT_BUFFER_M = 0.5
_LAT_BUFFER_PATH_M = 1.0

_BEV_X_RANGE = (0, 150)
_BEV_Y_RANGE = (-40, 40)
_BEV_SCALE = 6  # pixels per meter — large BEV panels


# --- geometry & radar ----------------------------------------------------------

def radar_spherical_to_cartesian(pts):
    az = pts["azimuth_angle"].astype(np.float64)
    el = pts["elevation_angle"].astype(np.float64)
    rg = pts["radar_range"].astype(np.float64)
    x = rg * np.cos(el) * np.cos(az)
    y = rg * np.cos(el) * np.sin(az)
    z = rg * np.sin(el)
    return x, y, z


def path_points_from_curvature(curvature_inv_m: float, max_dist: float = 100, n_pts: int = 100):
    k = curvature_inv_m
    if abs(k) < 1e-6:
        return [(s, 0.0) for s in np.linspace(0, max_dist, n_pts)]
    R = 1.0 / k
    return [(R * np.sin(k * s), R * (1 - np.cos(k * s))) for s in np.linspace(0, max_dist, n_pts)]


def pixel_to_h_angle_deg(u: float, W: float, H: float, hfov_deg: float) -> float:
    return ((u - W / 2) / (W / 2)) * (hfov_deg / 2)


def cam_dir_to_radar_azimuth(h_angle_deg, cam_ext, radar_ext):
    h_rad = np.deg2rad(h_angle_deg)
    dir_cam = np.array([np.sin(h_rad), 0.0, np.cos(h_rad)])
    R_cam = np.array(cam_ext)[:3, :3]
    R_radar = np.array(radar_ext)[:3, :3]
    dir_world = R_cam @ dir_cam
    dir_radar = R_radar.T @ dir_world
    return float(np.arctan2(dir_radar[1], dir_radar[0]))


def _polar_vel_dist(a, b, range_scale=4.0, lat_buffer=0.5, vel_scale=1.5):
    dr = abs(a[0] - b[0])
    r_avg = (a[0] + b[0]) / 2
    daz = abs(np.angle(np.exp(1j * (a[1] - b[1]))))
    d_lateral = r_avg * abs(np.sin(daz)) if r_avg > 0 else 0.0
    dv = abs(a[2] - b[2])
    return np.sqrt((dr / range_scale) ** 2 + (d_lateral / lat_buffer) ** 2 + (dv / vel_scale) ** 2)


def get_radar_xy_and_clusters(radar_data, ts_ns, z_min=-0.5, z_max=1.0, range_scale=4.0, lat_buffer=0.5, vel_scale=1.5, min_samples=2):
    pts = radar_data[radar_data["timestamp"] == ts_ns]
    if len(pts) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=int), pts, []
    x, y, z = radar_spherical_to_cartesian(pts)
    mask = (z >= z_min) & (z <= z_max)
    x_f, y_f = x[mask], y[mask]
    xy = np.column_stack([x_f, y_f])
    pts_f = pts[mask]
    if len(xy) == 0 or DBSCAN is None:
        return xy, np.zeros(len(xy), dtype=int), pts_f, []

    rg = pts_f["radar_range"].astype(np.float64)
    az = pts_f["azimuth_angle"].astype(np.float64)
    rr = pts_f["range_rate"].astype(np.float64)
    polar_vel = np.column_stack([rg, az, rr])
    metric = lambda a, b: _polar_vel_dist(a, b, range_scale, lat_buffer, vel_scale)
    labels = DBSCAN(eps=1.0, min_samples=min_samples, metric=metric).fit(polar_vel).labels_
    clusters = []
    for lbl in set(labels):
        if lbl < 0:
            continue
        m = labels == lbl
        clusters.append({
            "azimuth": float(np.mean(pts_f["azimuth_angle"][m])),
            "range": float(np.mean(pts_f["radar_range"][m])),
            "range_rate": float(np.mean(pts_f["range_rate"][m])),
            "indices": np.where(m)[0],
        })
    return xy, labels, pts_f, clusters


def _path_azimuth_at_range(curvature_inv_m: float, range_m: float) -> float:
    k = curvature_inv_m
    if abs(k) < 1e-9:
        return 0.0
    R = 1.0 / k
    r = min(range_m, 2 * R - 1e-6)
    theta = 2 * np.arcsin(r / (2 * R))
    x = R * np.sin(theta)
    y = R * (1 - np.cos(theta))
    return float(np.arctan2(y, x))


def find_cluster_on_path_direct(
    radar_data, ts_ns, curvature_inv_m, pts_f_ref, lat_buffer_m=1.0,
    z_min=-0.5, z_max=1.0, range_gap_m=4.0, vel_gap_ms=3.0, min_pts=2,
):
    if len(pts_f_ref) == 0:
        return None, None
    rg = pts_f_ref["radar_range"].astype(np.float64)
    az = pts_f_ref["azimuth_angle"].astype(np.float64)
    rr = pts_f_ref["range_rate"].astype(np.float64)
    on_path = []
    for i in range(len(pts_f_ref)):
        az_path = _path_azimuth_at_range(curvature_inv_m, rg[i])
        daz = abs(np.angle(np.exp(1j * (az[i] - az_path))))
        d_lateral = rg[i] * abs(np.sin(daz))
        if d_lateral <= lat_buffer_m:
            on_path.append((i, float(rg[i]), float(az[i]), float(rr[i]), float(d_lateral)))
    if not on_path:
        return None, None
    on_path.sort(key=lambda p: p[1])
    groups = [[on_path[0]]]
    for pt in on_path[1:]:
        last = groups[-1][-1]
        if abs(pt[1] - last[1]) <= range_gap_m and abs(pt[3] - last[3]) <= vel_gap_ms:
            groups[-1].append(pt)
        else:
            groups.append([pt])
    best, best_indices, best_score = None, None, (float("inf"), float("inf"))
    for group in groups:
        if len(group) < min_pts:
            continue
        indices = [p[0] for p in group]
        mean_dlat = float(np.mean([p[4] for p in group]))
        mean_range = float(np.mean([p[1] for p in group]))
        score = (mean_dlat, mean_range)
        if score < best_score:
            best_score = score
            best_indices = indices
            best = {
                "range": mean_range,
                "azimuth": float(np.mean([p[2] for p in group])),
                "range_rate": float(np.mean([p[3] for p in group])),
            }
    return best, best_indices


def find_nearest_cluster_lateral(clusters, azimuth_radar, lat_buffer_m=0.5):
    if not clusters:
        return None, -1
    in_cone = []
    for i, c in enumerate(clusters):
        daz = abs(np.angle(np.exp(1j * (c["azimuth"] - azimuth_radar))))
        d_lateral = c["range"] * abs(np.sin(daz))
        if d_lateral <= lat_buffer_m:
            in_cone.append((i, c))
    if not in_cone:
        return None, -1
    best_idx = min(in_cone, key=lambda x: x[1]["range"])[0]
    return clusters[best_idx], best_idx


# --- BEV drawing --------------------------------------------------------------

def draw_bev_raw(xy, scale=_BEV_SCALE, x_range=_BEV_X_RANGE, y_range=_BEV_Y_RANGE):
    """All radar returns only — grid, gray points, ego. 0–150 m forward."""
    x_range = x_range or _BEV_X_RANGE
    y_range = y_range or _BEV_Y_RANGE
    bev_h = int((x_range[1] - x_range[0]) * scale)
    bev_w = int((y_range[1] - y_range[0]) * scale)
    bev = np.ones((bev_h, bev_w, 3), dtype=np.uint8) * 28

    def to_pixel(x, y):
        row = int((x_range[1] - x) * scale)
        col = int((y_range[1] - y) * scale)
        return np.clip(row, 0, bev_h - 1), np.clip(col, 0, bev_w - 1)

    for x in range(0, int(x_range[1]) + 1, 25):
        r, c = to_pixel(x, y_range[0])
        cv2.line(bev, (c, r), (bev_w - 1, r), (55, 55, 55), 1)
        cv2.putText(bev, f"{x}m", (5, r + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)
    for y in range(y_range[0], y_range[1] + 1, 20):
        r, c = to_pixel(x_range[0], y)
        cv2.line(bev, (c, 0), (c, bev_h - 1), (55, 55, 55), 1)

    for i in range(len(xy)):
        r, c = to_pixel(xy[i, 0], xy[i, 1])
        cv2.circle(bev, (c, r), 2, (85, 85, 85), -1)

    r0, c0 = to_pixel(0, 0)
    cv2.circle(bev, (c0, r0), 8, (0, 255, 255), -1)
    cv2.circle(bev, (c0, r0), 10, (255, 255, 255), 2)

    cv2.putText(bev, "Raw radar — all points (z in [-0.5,1] m)", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(bev, f"0–{x_range[1]} m forward", (5, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return bev


def draw_bev_with_labeled_dot(
    xy, labeled_x, labeled_y, scale=_BEV_SCALE, x_range=None, y_range=None,
    curvature_inv_m=None, radar_bev_xy=None,
):
    x_range = x_range or _BEV_X_RANGE
    y_range = y_range or _BEV_Y_RANGE
    bev_h = int((x_range[1] - x_range[0]) * scale)
    bev_w = int((y_range[1] - y_range[0]) * scale)
    bev = np.ones((bev_h, bev_w, 3), dtype=np.uint8) * 28

    def to_pixel(x, y):
        row = int((x_range[1] - x) * scale)
        col = int((y_range[1] - y) * scale)
        return np.clip(row, 0, bev_h - 1), np.clip(col, 0, bev_w - 1)

    for x in range(0, int(x_range[1]) + 1, 25):
        r, c = to_pixel(x, y_range[0])
        cv2.line(bev, (c, r), (bev_w - 1, r), (55, 55, 55), 1)
        cv2.putText(bev, f"{x}m", (5, r + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
    for y in range(y_range[0], y_range[1] + 1, 20):
        r, c = to_pixel(x_range[0], y)
        cv2.line(bev, (c, 0), (c, bev_h - 1), (55, 55, 55), 1)

    for i in range(len(xy)):
        r, c = to_pixel(xy[i, 0], xy[i, 1])
        cv2.circle(bev, (c, r), 2, (85, 85, 85), -1)

    if curvature_inv_m is not None:
        path_pts = path_points_from_curvature(curvature_inv_m)
        for i in range(len(path_pts) - 1):
            r1, c1 = to_pixel(path_pts[i][0], path_pts[i][1])
            r2, c2 = to_pixel(path_pts[i + 1][0], path_pts[i + 1][1])
            cv2.line(bev, (c1, r1), (c2, r2), (0, 255, 0), 2)
        cv2.putText(bev, f"path k={curvature_inv_m:.3f}", (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if labeled_x is not None and labeled_y is not None and not (np.isnan(labeled_x) or np.isnan(labeled_y)):
        r, c = to_pixel(labeled_x, labeled_y)
        cv2.circle(bev, (c, r), 3, (0, 255, 255), -1)
        cv2.circle(bev, (c, r), 4, (255, 255, 255), 1)

    if radar_bev_xy is not None and len(radar_bev_xy) >= 2:
        rx, ry = float(radar_bev_xy[0]), float(radar_bev_xy[1])
        if not (np.isnan(rx) or np.isnan(ry)):
            r, c = to_pixel(rx, ry)
            cv2.circle(bev, (c, r), 3, (255, 255, 0), -1)
            cv2.circle(bev, (c, r), 4, (255, 255, 255), 1)

    r0, c0 = to_pixel(0, 0)
    cv2.circle(bev, (c0, r0), 8, (0, 255, 255), -1)
    cv2.circle(bev, (c0, r0), 10, (255, 255, 255), 2)

    cv2.putText(bev, "Labels BEV (GT check)", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    leg = "gray=radar  yellow=label  cyan=CIPO pt  green=path"
    cv2.putText(bev, leg, (5, bev_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    return bev


def draw_bev_association_fov(
    xy,
    matched_indices,
    az_radar_deg,
    scale=_BEV_SCALE,
    x_range=None,
    y_range=None,
    curvature_inv_m=None,
    path_only=False,
):
    """
    BEV for BR panel: radar points, ego path, association — **no** lateral-buffer polygons.
    Azimuth: single yellow line along the CIPO/path ray (not dotted “fox” corridor).
    """
    x_range = x_range or _BEV_X_RANGE
    y_range = y_range or _BEV_Y_RANGE
    bev_h = int((x_range[1] - x_range[0]) * scale)
    bev_w = int((y_range[1] - y_range[0]) * scale)
    bev = np.ones((bev_h, bev_w, 3), dtype=np.uint8) * 28

    def to_pixel(x, y):
        row = int((x_range[1] - x) * scale)
        col = int((y_range[1] - y) * scale)
        return np.clip(row, 0, bev_h - 1), np.clip(col, 0, bev_w - 1)

    for x in range(0, int(x_range[1]) + 1, 25):
        r, c = to_pixel(x, y_range[0])
        cv2.line(bev, (c, r), (bev_w - 1, r), (55, 55, 55), 1)
        cv2.putText(bev, f"{x}m", (5, r + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
    for y in range(y_range[0], y_range[1] + 1, 20):
        r, c = to_pixel(x_range[0], y)
        cv2.line(bev, (c, 0), (c, bev_h - 1), (55, 55, 55), 1)

    for i in range(len(xy)):
        r, c = to_pixel(xy[i, 0], xy[i, 1])
        cv2.circle(bev, (c, r), 2, (85, 85, 85), -1)

    if curvature_inv_m is not None:
        path_pts = path_points_from_curvature(curvature_inv_m)
        for i in range(len(path_pts) - 1):
            r1, c1 = to_pixel(path_pts[i][0], path_pts[i][1])
            r2, c2 = to_pixel(path_pts[i + 1][0], path_pts[i + 1][1])
            cv2.line(bev, (c1, r1), (c2, r2), (0, 255, 0), 2)
        cv2.putText(bev, f"path k={curvature_inv_m:.3f}", (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if not path_only:
        az_rad = np.deg2rad(az_radar_deg)
        r0, c0 = to_pixel(0, 0)
        r1, c1 = to_pixel(120 * np.cos(az_rad), 120 * np.sin(az_rad))
        cv2.line(bev, (c0, r0), (c1, r1), (0, 220, 255), 3)

    if matched_indices is not None:
        for i in matched_indices:
            r, c = to_pixel(xy[i, 0], xy[i, 1])
            cv2.circle(bev, (c, r), 7, (0, 255, 0), -1)
            cv2.circle(bev, (c, r), 9, (255, 255, 255), 2)

    r_ego, c_ego = to_pixel(0, 0)
    cv2.circle(bev, (c_ego, r_ego), 8, (0, 255, 255), -1)
    cv2.circle(bev, (c_ego, r_ego), 10, (255, 255, 255), 2)

    cv2.putText(bev, "Assoc BEV (FOV ray, no buffer)", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    if path_only:
        cv2.putText(bev, "path + moving radar", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 200, 200), 1)
    else:
        cv2.putText(bev, f"az {az_radar_deg:.1f} deg", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)
    return bev


def render_fov_cipo_camera(img_bgr, preds, W: int, H: int):
    """
    BR top strip: camera FOV wedge only (no class bboxes / scores). CIPO foot point: bold circle if L1/L2 exists.
    """
    img = img_bgr.copy()
    cx = W // 2
    apex_y = H - 12
    top_y = max(24, int(H * 0.22))
    # Horizontal FOV guide: left / center / right from near bumper
    cv2.line(img, (cx, apex_y), (0, top_y), (140, 190, 255), 2, cv2.LINE_AA)
    cv2.line(img, (cx, apex_y), (W - 1, top_y), (140, 190, 255), 2, cv2.LINE_AA)
    cv2.line(img, (cx, apex_y), (cx, top_y), (90, 90, 110), 1, cv2.LINE_AA)

    CIPO_CLASSES = (1, 2)
    cipo = [p for p in preds if int(p[5]) in CIPO_CLASSES]
    if cipo:
        cipo.sort(key=lambda p: (p[1] + p[3]) / 2, reverse=True)
        x1, y1, x2, y2, _, _ = cipo[0]
        u, v = int((x1 + x2) / 2), int(y2)
        cv2.circle(img, (u, v), 22, (0, 255, 0), 4)
        cv2.circle(img, (u, v), 10, (0, 255, 255), -1)
        cv2.circle(img, (u, v), 22, (255, 255, 255), 2)

    cv2.putText(img, "Camera: FOV wedge + CIPO contact", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1)
    return img


def resize_fit(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Resize keeping aspect ratio to fit inside max_w x max_h."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def pad_to_size(img: np.ndarray, tw: int, th: int, fill=(20, 20, 20)) -> np.ndarray:
    h, w = img.shape[:2]
    out = np.full((th, tw, 3), fill, dtype=np.uint8)
    y0 = (th - h) // 2
    x0 = (tw - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = img
    return out


def render_labels_camera(img_bgr, cipo_rec, label: dict, seq: str, stem: str) -> np.ndarray:
    """Camera strip with label overlays (debug_labels style)."""
    img = img_bgr.copy()
    if cipo_rec.get("cipo_detected") and "bbox" in cipo_rec:
        x1, y1, x2, y2 = [int(v) for v in cipo_rec["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "CIPO", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    px = cipo_rec.get("pixel_point")
    if px is not None and len(px) >= 2:
        cv2.circle(img, (int(px[0]), int(px[1])), 6, (0, 255, 255), 2)

    curvature = label.get("curvature")
    dist = label.get("distance_to_in_path_object")
    speed = label.get("speed_of_in_path_object")
    curv_str = f"{curvature:.4f}" if curvature is not None and isinstance(curvature, (int, float)) else "-"

    cipo_dist = cipo_rec.get("distance_m")
    cipo_speed_adj = cipo_rec.get("speed_ms_adjusted")
    lines = [
        "LABELS (camera)",
        f"k: {curv_str}",
        f"dist: {dist} m" if dist is not None else "dist: -",
        f"spd: {speed}" if speed is not None else "spd: -",
    ]
    if cipo_dist is not None or cipo_speed_adj is not None:
        t = f"radar: {cipo_dist}m" if cipo_dist is not None else "radar: -"
        if cipo_speed_adj is not None:
            t += f"  v_adj={cipo_speed_adj:.1f}"
        lines.append(t)

    cv2.rectangle(img, (0, 0), (img.shape[1], 32 + len(lines) * 28), (40, 40, 40), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (16, 28 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    suf = " (prev)" if cipo_rec.get("track_from_prev") else (" (nb)" if cipo_rec.get("track_from_neighbor") else (" (path)" if cipo_rec.get("cipo_from_path") else ""))
    cv2.putText(img, f"{seq}  {stem}{suf}", (16, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return img


# --- main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="2x2 ZOD debug grid (raw radar | labels | cipo)")
    parser.add_argument("--sequence", type=str, default="000479")
    parser.add_argument("--zod-root", type=str, required=True)
    parser.add_argument("--every", type=int, default=20, help="Process every Nth frame")
    parser.add_argument("--labels-dir", type=str, default=None)
    parser.add_argument("--cipo-radar", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="AutoSpeed checkpoint (default: {zod_root}/models/autospeed.pt)",
    )
    parser.add_argument("--cell-max", type=int, default=1100, help="Max width/height per quadrant (px)")
    args = parser.parse_args()

    from zod_utils import default_autospeed_checkpoint, get_calibration_path, get_images_blur_dir, sequence_output_dir

    zod = Path(args.zod_root)
    seq = args.sequence
    labels_dir = Path(args.labels_dir) if args.labels_dir else zod / "labels" / seq
    cipo_path = Path(args.cipo_radar) if args.cipo_radar else sequence_output_dir(zod, seq) / "cipo_radar.json"
    out_dir = Path(args.output_dir) if args.output_dir else sequence_output_dir(zod, seq) / "debug" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path) if args.model_path else default_autospeed_checkpoint(zod)

    img_dir = get_images_blur_dir(zod, seq)
    calib_path = get_calibration_path(zod, seq)
    assoc_path = zod / "associations" / f"{seq}_associations.json"

    if not assoc_path.exists():
        print(f"Missing associations: {assoc_path}")
        return 1
    if not calib_path.exists():
        print(f"Missing calibration: {calib_path}")
        return 1
    if not cipo_path.exists():
        print(f"Missing cipo_radar: {cipo_path}")
        return 1
    if not labels_dir.exists():
        print(f"Missing labels dir: {labels_dir}")
        return 1

    with open(assoc_path) as f:
        assoc = json.load(f)
    with open(calib_path) as f:
        calib = json.load(f)["FC"]
    with open(cipo_path) as f:
        cipo_data = json.load(f)

    W, H = calib["image_dimensions"][0], calib["image_dimensions"][1]
    hfov_deg = calib["field_of_view"][0]
    cam_ext = np.array(calib["extrinsics"])
    radar_ext = np.array(calib["radar_extrinsics"])

    radar_data = np.load(assoc["radar_npy_path"], allow_pickle=True)
    cipo_map = {r["image"]: r for r in cipo_data["results"]}
    model = AutoSpeedNetworkInfer(str(model_path))

    cm = args.cell_max
    n_saved = 0
    CIPO_CLASSES = (1, 2)

    for idx, rec in enumerate(assoc["associations"]):
        if idx % args.every != 0:
            continue
        img_name = rec["image"]
        stem = Path(img_name).stem
        img_path = img_dir / img_name
        label_path = labels_dir / f"{stem}.json"
        if not img_path.exists() or not label_path.exists():
            continue

        with open(label_path) as f:
            label = json.load(f)
        cipo_rec = cipo_map.get(img_name, {})

        img_pil = Image.open(img_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        preds = model.inference(img_pil)

        pts = radar_data[radar_data["timestamp"] == rec["radar_timestamp_ns"]]
        x, y, z = radar_spherical_to_cartesian(pts)
        mask = (z >= -0.5) & (z <= 1.0)
        xy = np.column_stack([x[mask], y[mask]])

        panel_tl = draw_bev_raw(xy, scale=_BEV_SCALE)

        dist = label.get("distance_to_in_path_object")
        az_deg = cipo_rec.get("azimuth_radar_deg")
        labeled_x = labeled_y = None
        if dist is not None and az_deg is not None:
            if not (np.isnan(dist) or np.isnan(az_deg)):
                az_rad = np.deg2rad(az_deg)
                labeled_x = float(dist * np.cos(az_rad))
                labeled_y = float(dist * np.sin(az_rad))
        curv = label.get("curvature") or rec.get("curvature_inv_m")
        panel_tr = draw_bev_with_labeled_dot(
            xy, labeled_x, labeled_y,
            curvature_inv_m=curv,
            radar_bev_xy=cipo_rec.get("bev_xy"),
        )

        panel_bl = render_labels_camera(img_bgr, cipo_rec, label, seq, stem)

        xy_c, labels_c, pts_f, clusters = get_radar_xy_and_clusters(
            radar_data, rec["radar_timestamp_ns"], lat_buffer=_LAT_BUFFER_M
        )
        cipo = [p for p in preds if int(p[5]) in CIPO_CLASSES]
        if not cipo:
            curvature = rec.get("curvature_inv_m", 0.0)
            cluster_path, matched_path = find_cluster_on_path_direct(
                radar_data, rec["radar_timestamp_ns"], curvature, pts_f, lat_buffer_m=_LAT_BUFFER_PATH_M,
            )
            bev_cipo = draw_bev_association_fov(
                xy_c, matched_path, 0.0,
                curvature_inv_m=curvature, path_only=True,
            )
            cam_cipo = render_fov_cipo_camera(img_bgr, preds, W, H)
        else:
            cipo.sort(key=lambda p: (p[1] + p[3]) / 2, reverse=True)
            x1, y1, x2, y2, _, _ = cipo[0]
            u = (x1 + x2) / 2
            h_angle_deg = pixel_to_h_angle_deg(u, W, H, hfov_deg)
            az_radar = cam_dir_to_radar_azimuth(h_angle_deg, cam_ext, radar_ext)
            az_radar_deg = float(np.rad2deg(az_radar))
            cluster, _ = find_nearest_cluster_lateral(clusters, az_radar, lat_buffer_m=_LAT_BUFFER_M)
            matched_indices = cluster["indices"].tolist() if cluster else None
            curvature = rec.get("curvature_inv_m")
            bev_cipo = draw_bev_association_fov(
                xy_c, matched_indices, az_radar_deg,
                curvature_inv_m=curvature, path_only=False,
            )
            cam_cipo = render_fov_cipo_camera(img_bgr, preds, W, H)

        # BR quadrant: CIPO camera (top) + CIPO BEV (bottom) → resize to cm×cm
        hh = max(1, cm // 2)
        cam_s = pad_to_size(resize_fit(cam_cipo, cm, hh), cm, hh)
        bev_s = pad_to_size(resize_fit(bev_cipo, cm, hh), cm, hh)
        panel_br = np.vstack([cam_s, bev_s])
        panel_br = cv2.resize(panel_br, (cm, cm), interpolation=cv2.INTER_AREA)

        panel_tl = cv2.resize(resize_fit(panel_tl, cm, cm), (cm, cm), interpolation=cv2.INTER_AREA)
        panel_tr = cv2.resize(resize_fit(panel_tr, cm, cm), (cm, cm), interpolation=cv2.INTER_AREA)
        panel_bl = cv2.resize(resize_fit(panel_bl, cm, cm), (cm, cm), interpolation=cv2.INTER_AREA)

        top = np.hstack([panel_tl, panel_tr])
        bot = np.hstack([panel_bl, panel_br])
        canvas = np.vstack([top, bot])

        banner_h = 44
        banner = np.full((banner_h, canvas.shape[1], 3), 32, dtype=np.uint8)
        title = f"ZOD 2x2  seq {seq}  frame {idx}  {stem}   | TL raw  TR labels BEV  BL labels cam  BR FOV+CIPO / assoc BEV"
        cv2.putText(banner, title[: min(180, len(title))], (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)
        canvas = np.vstack([banner, canvas])

        out_path = out_dir / f"grid_{idx:04d}_{stem}.png"
        cv2.imwrite(str(out_path), canvas)
        n_saved += 1
        print(f"Saved {out_path.name}")

    print(f"Done: {n_saved} grids -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
