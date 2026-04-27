import os
os.environ["OMP_NUM_THREADS"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs_web")

import math
import glob
import traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import gradio as gr

# ============================
# 路径与参数配置区（只改这里）
# ============================
WEIGHTS_ROI = "/root/ultralytics-main/best.pt"
WEIGHTS_LINE = "/root/ultralytics-main/runs/detect/y11m_mix_img1280_b16_py2/weights/best.pt"
REAL_ESRGAN_MODEL_PATH = "/root/autodl-tmp/Real-ESRGAN/weights/RealESRGAN_x4plus.pthj"
INPUT_DIR = "/root/ultralytics-main/dataset/image"
OUTPUT_ROOT = "/root/app/outputs_web"
STANDARD_SPEC_CSV = "/root/ultralytics-main/specs/standard_spec.csv"

ENABLE_SR = True
SR_SCALE = 2
SR_RESIZE_BACK = False

ROI_CONF = 0.25
ROI_IOU = 0.60
ROI_PAD_RATIO = 0.10

LINE_CONF = 0.10
LINE_IOU = 0.60
LINE_IMGSZ = 1280
DEVICE = 0 if torch.cuda.is_available() else "cpu"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

COLOR_PASS = (0, 255, 0)
COLOR_FAIL = (0, 0, 255)
COLOR_ROI_ID = (0, 255, 255)

# ============================
# 尺寸标定参数
# ============================
REAL_UNIT = "mm"
KNOWN_CALIBRATION_LENGTH_MM = 5.0   # 你已知参考物真实长度 = 5 mm

# ============================
# 线条测量参数
# ============================
DEDUP_IOU = 0.25
CENTER_DIST_RATIO = 0.03
Y_CENTER_RATIO = 0.015
HEIGHT_RATIO_DIFF = 0.6
X_OVERLAP_RATIO = 0.5

BODY_Y1_RATIO = 0.10
BODY_Y2_RATIO = 0.90
BODY_EDGE_SMOOTH_SIGMA = 7.0

BAND_HALF = 7
MAX_GAP = 4
INNER_SEARCH_PAD = 45
OUTER_SNAP_INSET = 0
MIN_SEG_LEN_RATIO = 0.15
INNER_THR_RATIO = 0.58

EDGE_SEARCH_MARGIN = 12
EDGE_GRAD_SMOOTH_SIGMA = 1.2
EDGE_SCORE_RATIO = 0.45

REFINE_Y_PAD = 6
Y_THR_RATIO = 0.60
MIN_BOX_H = 4

MIN_VERTICAL_GAP = 2
SEPARATOR_ROW_SMOOTH_SIGMA = 1.0
SEPARATOR_SEARCH_PAD = 2
OVERLAP_RESOLVE_PASSES = 2

ROI_PAD_X = 2
ROI_PAD_Y = 2
CENTER_BAND_RATIO = 0.60
MIN_CENTER_BAND_H = 8
ROW_PERCENTILE = 25
ROW_SMOOTH_SIGMA = 1.2
ROW_THR_RATIO = 0.40
FALLBACK_HEIGHT_RATIO = 0.35
MIN_MASK_AREA = 10
PARALLEL_ANGLE_THR_DEG = 2.0

DRAW_BODY_EDGE = True
DRAW_SIDE_TEXT = True
DRAW_CONTOUR = True
DRAW_CENTERLINE = True
DRAW_WIDTH_LINE = True
DRAW_LENGTH_TEXT = True
DRAW_WIDTH_TEXT = True
DRAW_PARALLEL_TEXT = True

WEB_HOST = "0.0.0.0"
WEB_PORT = 7860
WEB_SHARE = False

# ============================
# 可选 SR 依赖
# ============================
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    HAS_REAL_ESRGAN = True
except Exception:
    HAS_REAL_ESRGAN = False
    RealESRGANer = None
    RRDBNet = None


@dataclass
class RoiMeta:
    image_name: str
    roi_name: str
    roi_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float


@dataclass
class LineMetric:
    line_index: int
    side: str
    cls: int
    conf: float
    box_x1: float
    box_y1: float
    box_x2: float
    box_y2: float
    length_px: float
    width_px: float
    length_real_mm: Optional[float]
    width_real_mm: Optional[float]
    angle_deg: float
    parallel_diff_next_deg: Optional[float]
    is_parallel_next: Optional[str]
    next_gap_y: Optional[float]
    next_gap_real_mm: Optional[float]
    center_x: float
    center_y: float


@dataclass
class RoiResult:
    roi_meta: RoiMeta
    num_lines: int
    status: str
    fail_reasons: str
    line_metrics: List[LineMetric]
    vis_path: str
    sr_path: Optional[str]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def smooth_1d(arr, sigma=3.0):
    arr = np.asarray(arr, dtype=np.float32).reshape(1, -1)
    out = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma).reshape(-1)
    return out


def normalize_angle_180(angle_deg):
    angle = angle_deg % 180.0
    if angle < 0:
        angle += 180.0
    return angle


def angle_diff_deg(a1, a2):
    d = abs(a1 - a2) % 180.0
    return min(d, 180.0 - d)


def judge_parallel_yes_no(angle_diff, thr_deg=PARALLEL_ANGLE_THR_DEG):
    if angle_diff is None:
        return None
    return "yes" if angle_diff <= thr_deg else "no"


def enhance_gray(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=7)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def np_to_bgr(img_np: np.ndarray) -> np.ndarray:
    if img_np is None:
        raise ValueError("未接收到图像")
    if img_np.ndim == 2:
        return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def init_calibration_state():
    return {
        "points": [],              # [(x1, y1), (x2, y2)]
        "pixel_length": None,
        "known_length_mm": KNOWN_CALIBRATION_LENGTH_MM,
        "mm_per_px": None,
        "is_calibrated": False,
        "source": None,
    }


def draw_calibration_overlay(img_rgb, state):
    if img_rgb is None:
        return None
    canvas = img_rgb.copy()
    if not isinstance(state, dict):
        return canvas

    pts = state.get("points", [])
    for i, (x, y) in enumerate(pts):
        cv2.circle(canvas, (int(x), int(y)), 6, (255, 0, 0), -1)
        cv2.putText(canvas, f"P{i+1}", (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[0], pts[1]
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        px_len = state.get("pixel_length", None)
        mm_per_px = state.get("mm_per_px", None)
        text_y = max(25, min(int((y1 + y2) / 2) - 10, canvas.shape[0] - 10))
        text_x = max(10, min(int((x1 + x2) / 2), canvas.shape[1] - 220))
        msg = ""
        if px_len is not None:
            msg += f"px={px_len:.2f}"
        if mm_per_px is not None:
            msg += f", mm/px={mm_per_px:.6f}"
        if msg:
            cv2.putText(canvas, msg, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    return canvas


class SuperResolutionEngine:
    def __init__(self, model_path: str, enable_sr: bool = True, sr_scale: int = 2):
        self.enable_sr = enable_sr and HAS_REAL_ESRGAN and os.path.exists(model_path)
        self.sr_scale = sr_scale
        self.upsampler = None
        if self.enable_sr:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23,
                num_grow_ch=32, scale=4
            )
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=torch.cuda.is_available()
            )

    def run(self, bgr: np.ndarray) -> np.ndarray:
        if not self.enable_sr or self.upsampler is None:
            return bgr.copy()
        out, _ = self.upsampler.enhance(bgr, outscale=self.sr_scale)
        if SR_RESIZE_BACK:
            h, w = bgr.shape[:2]
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
        return out


class RoiDetector:
    def __init__(self, weights: str):
        self.model = YOLO(weights)

    def detect(self, img_bgr: np.ndarray, image_name: str) -> List[RoiMeta]:
        H, W = img_bgr.shape[:2]
        result = self.model.predict(source=img_bgr, conf=ROI_CONF, iou=ROI_IOU, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        idx = sorted(range(len(boxes)), key=lambda i: (boxes[i][0] + boxes[i][2]) / 2.0)

        rois = []
        for roi_id, i in enumerate(idx, start=1):
            x1, y1, x2, y2 = boxes[i]
            c = float(confs[i])
            bw, bh = (x2 - x1), (y2 - y1)
            pad = ROI_PAD_RATIO * max(bw, bh)
            xx1 = int(clip(x1 - pad, 0, W - 1))
            yy1 = int(clip(y1 - pad, 0, H - 1))
            xx2 = int(clip(x2 + pad, 0, W - 1))
            yy2 = int(clip(y2 + pad, 0, H - 1))
            roi_name = f"{os.path.splitext(image_name)[0]}_roi{roi_id:03d}.png"
            rois.append(RoiMeta(image_name, roi_name, roi_id, xx1, yy1, xx2, yy2, c))
        return rois


class LineMeasurementEngine:
    def __init__(self, weights: str):
        self.model = YOLO(weights)

    def box_iou_xyxy(self, box1, box2):
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2
        inter_x1 = max(x11, x21)
        inter_y1 = max(y11, y21)
        inter_x2 = min(x12, x22)
        inter_y2 = min(y12, y22)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
        area2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
        union = area1 + area2 - inter_area
        return 0.0 if union <= 0 else inter_area / union

    def is_duplicate_box(self, candidate, kept, img_w, img_h):
        b1 = [candidate["x1"], candidate["y1"], candidate["x2"], candidate["y2"]]
        b2 = [kept["x1"], kept["y1"], kept["x2"], kept["y2"]]
        iou = self.box_iou_xyxy(b1, b2)
        if iou >= DEDUP_IOU:
            return True
        cx1 = 0.5 * (candidate["x1"] + candidate["x2"])
        cy1 = 0.5 * (candidate["y1"] + candidate["y2"])
        cx2 = 0.5 * (kept["x1"] + kept["x2"])
        cy2 = 0.5 * (kept["y1"] + kept["y2"])
        center_dist = np.hypot(cx1 - cx2, cy1 - cy2)
        if center_dist <= CENTER_DIST_RATIO * max(img_w, img_h):
            return True
        h1 = max(1.0, candidate["y2"] - candidate["y1"])
        h2 = max(1.0, kept["y2"] - kept["y1"])
        y_close = abs(cy1 - cy2) <= Y_CENTER_RATIO * img_h
        h_similar = abs(h1 - h2) / max(h1, h2) <= HEIGHT_RATIO_DIFF
        x_left = max(candidate["x1"], kept["x1"])
        x_right = min(candidate["x2"], kept["x2"])
        x_overlap = max(0.0, x_right - x_left)
        w1 = max(1.0, candidate["x2"] - candidate["x1"])
        w2 = max(1.0, kept["x2"] - kept["x1"])
        x_overlap_ratio = x_overlap / min(w1, w2)
        return y_close and h_similar and x_overlap_ratio >= X_OVERLAP_RATIO

    def deduplicate_boxes(self, preds, img_w, img_h):
        preds = sorted(preds, key=lambda d: d["conf"], reverse=True)
        kept = []
        for p in preds:
            duplicated = False
            for k in kept:
                if self.is_duplicate_box(p, k, img_w, img_h):
                    duplicated = True
                    break
            if not duplicated:
                kept.append(p)
        return kept

    def fill_small_gaps_1d(self, mask, max_gap=6):
        mask = mask.astype(np.uint8).copy()
        n = len(mask)
        i = 0
        while i < n:
            if mask[i] == 0:
                j = i
                while j < n and mask[j] == 0:
                    j += 1
                if i > 0 and j < n and (j - i) <= max_gap:
                    mask[i:j] = 1
                i = j
            else:
                i += 1
        return mask

    def find_runs(self, mask):
        runs = []
        n = len(mask)
        i = 0
        while i < n:
            if mask[i] == 1:
                j = i
                while j < n and mask[j] == 1:
                    j += 1
                runs.append((i, j - 1))
                i = j
            else:
                i += 1
        return runs

    def sort_by_y(self, preds):
        for p in preds:
            p["cy"] = 0.5 * (p["y1"] + p["y2"])
        return sorted(preds, key=lambda d: d["cy"])

    def robust_dark_profile(self, roi):
        if roi.size == 0:
            return None
        q = np.percentile(roi, 30, axis=0).astype(np.float32)
        return 255.0 - q

    def estimate_resistor_body_x(self, gray):
        H, W = gray.shape[:2]
        y1 = int(round(H * BODY_Y1_RATIO))
        y2 = int(round(H * BODY_Y2_RATIO))
        y1 = max(0, min(H - 2, y1))
        y2 = max(y1 + 1, min(H - 1, y2))
        band = gray[y1:y2, :]
        if band.size == 0:
            return 0, W - 1
        blur = cv2.GaussianBlur(band, (0, 0), 2.0)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        pos = np.mean(np.clip(gx, 0, None), axis=0)
        neg = np.mean(np.clip(-gx, 0, None), axis=0)
        pos = smooth_1d(pos, BODY_EDGE_SMOOTH_SIGMA)
        neg = smooth_1d(neg, BODY_EDGE_SMOOTH_SIGMA)
        l_lo, l_hi = int(0.05 * W), int(0.45 * W)
        r_lo, r_hi = int(0.55 * W), int(0.95 * W)
        left_x = l_lo + int(np.argmax(pos[l_lo:l_hi])) if l_hi > l_lo else 0
        right_x = r_lo + int(np.argmax(neg[r_lo:r_hi])) if r_hi > r_lo else W - 1
        if right_x <= left_x + 20:
            col_mean = np.mean(blur, axis=0).astype(np.float32)
            col_mean = smooth_1d(col_mean, BODY_EDGE_SMOOTH_SIGMA)
            thr = np.percentile(col_mean, 60)
            mask = (col_mean >= thr).astype(np.uint8)
            mask = self.fill_small_gaps_1d(mask, max_gap=15)
            runs = self.find_runs(mask)
            if runs:
                center = 0.5 * W
                best, best_score = None, -1e9
                for l, r in runs:
                    w = r - l + 1
                    c = 0.5 * (l + r)
                    score = w - 0.5 * abs(c - center)
                    if score > best_score:
                        best_score = score
                        best = (l, r)
                if best is not None:
                    left_x, right_x = best
        left_x = int(np.clip(left_x, 0, W - 2))
        right_x = int(np.clip(right_x, left_x + 1, W - 1))
        return left_x, right_x

    def classify_half_side(self, box, body_x1, body_x2):
        d_left = abs(box["x1"] - body_x1)
        d_right = abs(body_x2 - box["x2"])
        if d_left < d_right:
            return "left"
        if d_right < d_left:
            return "right"
        body_cx = 0.5 * (body_x1 + body_x2)
        box_cx = 0.5 * (box["x1"] + box["x2"])
        return "left" if box_cx < body_cx else "right"

    def choose_best_run_for_box(self, runs, cur_l, cur_r):
        best = None
        best_overlap = -1
        for l, r in runs:
            overlap = max(0, min(r, cur_r) - max(l, cur_l) + 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best = (l, r)
        return best

    def locate_inner_edge_1d(self, local_score, core_l, core_r, side):
        if local_score is None or len(local_score) < 5:
            return None
        s = smooth_1d(local_score, EDGE_GRAD_SMOOTH_SIGMA)
        grad = np.gradient(s).astype(np.float32)
        if side == "left":
            search_l = max(1, core_r - EDGE_SEARCH_MARGIN)
            search_r = min(len(s) - 2, core_r + EDGE_SEARCH_MARGIN)
            if search_r <= search_l:
                return None
            g = grad[search_l:search_r + 1]
            peak = float(np.max(g))
            if peak <= 1e-6:
                return None
            thr = EDGE_SCORE_RATIO * peak
            idxs = np.where(g >= thr)[0]
            return search_l + int(idxs[0]) if len(idxs) else None
        else:
            search_l = max(1, core_l - EDGE_SEARCH_MARGIN)
            search_r = min(len(s) - 2, core_l + EDGE_SEARCH_MARGIN)
            if search_r <= search_l:
                return None
            g = grad[search_l:search_r + 1]
            valley = float(np.min(g))
            if abs(valley) <= 1e-6:
                return None
            thr = EDGE_SCORE_RATIO * abs(valley)
            idxs = np.where((-g) >= thr)[0]
            return search_l + int(idxs[-1]) if len(idxs) else None

    def refine_box_y(self, gray, x1, y1, x2, y2):
        H, W = gray.shape[:2]
        x1i = max(0, min(W - 1, int(round(x1))))
        x2i = max(x1i + 1, min(W - 1, int(round(x2))))
        y1i = max(0, min(H - 1, int(round(y1 - REFINE_Y_PAD))))
        y2i = max(y1i + 1, min(H - 1, int(round(y2 + REFINE_Y_PAD))))
        roi = gray[y1i:y2i + 1, x1i:x2i + 1]
        if roi.size == 0:
            return y1, y2
        row_q = np.percentile(roi, 30, axis=1).astype(np.float32)
        row_score = 255.0 - row_q
        row_score = smooth_1d(row_score, 1.0)
        peak = float(np.max(row_score))
        if peak <= 1e-6:
            return y1, y2
        thr = Y_THR_RATIO * peak
        mask = (row_score >= thr).astype(np.uint8)
        mask = self.fill_small_gaps_1d(mask, max_gap=2)
        runs = self.find_runs(mask)
        if not runs:
            return y1, y2
        best = max(runs, key=lambda t: t[1] - t[0] + 1)
        yy1 = y1i + best[0]
        yy2 = y1i + best[1]
        if yy2 - yy1 + 1 < MIN_BOX_H:
            return y1, y2
        return float(yy1), float(yy2)

    def refine_half_box(self, gray, box, body_x1, body_x2):
        H, W = gray.shape[:2]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        cy = int(round(0.5 * (y1 + y2)))
        ys = max(0, cy - BAND_HALF)
        ye = min(H - 1, cy + BAND_HALF)
        if ye <= ys:
            return box
        side = self.classify_half_side(box, body_x1, body_x2)
        band = gray[ys:ye + 1, :]
        if band.size == 0:
            return box
        out = dict(box)
        body_w = max(1.0, body_x2 - body_x1)
        min_seg_len = max(8, int(round(MIN_SEG_LEN_RATIO * body_w)))
        if side == "left":
            search_l = max(0, int(round(body_x1)))
            search_r = min(W - 1, int(round(x2 + INNER_SEARCH_PAD)))
            if search_r <= search_l + 3:
                out["x1"] = float(body_x1 + OUTER_SNAP_INSET)
                out["side"] = "left"
                return out
            roi = band[:, search_l:search_r + 1]
            local_score = self.robust_dark_profile(roi)
            if local_score is None or len(local_score) < 5:
                out["x1"] = float(body_x1 + OUTER_SNAP_INSET)
                out["side"] = "left"
                return out
            core_l = max(0, int(round(x1)) - search_l)
            core_r = min(len(local_score) - 1, int(round(x2)) - search_l)
            if core_r <= core_l:
                out["x1"] = float(body_x1 + OUTER_SNAP_INSET)
                out["side"] = "left"
                return out
            core = local_score[core_l:core_r + 1]
            core_max = float(np.max(core)) if len(core) > 0 else 0.0
            if core_max <= 1e-6:
                out["x1"] = float(body_x1 + OUTER_SNAP_INSET)
                out["side"] = "left"
                return out
            thr = INNER_THR_RATIO * core_max
            binary = (local_score >= thr).astype(np.uint8)
            binary = self.fill_small_gaps_1d(binary, MAX_GAP)
            runs = self.find_runs(binary)
            if not runs:
                out["x1"] = float(body_x1 + OUTER_SNAP_INSET)
                out["side"] = "left"
                return out
            chosen = self.choose_best_run_for_box(runs, core_l, core_r)
            if chosen is None:
                out["x1"] = float(body_x1 + OUTER_SNAP_INSET)
                out["side"] = "left"
                return out
            l, r = chosen
            edge = self.locate_inner_edge_1d(local_score, l, r, side="left")
            if edge is None:
                edge = r
            new_x1 = float(body_x1 + OUTER_SNAP_INSET)
            new_x2 = float(search_l + edge)
            if new_x2 - new_x1 < min_seg_len:
                new_x2 = float(search_l + r)
            if new_x2 <= new_x1 + 2:
                return box
            new_y1, new_y2 = self.refine_box_y(gray, new_x1, y1, new_x2, y2)
            out.update({"x1": new_x1, "x2": new_x2, "y1": new_y1, "y2": new_y2, "side": "left"})
            return out
        else:
            search_l = max(0, int(round(x1 - INNER_SEARCH_PAD)))
            search_r = min(W - 1, int(round(body_x2)))
            if search_r <= search_l + 3:
                out["x2"] = float(body_x2 - OUTER_SNAP_INSET)
                out["side"] = "right"
                return out
            roi = band[:, search_l:search_r + 1]
            local_score = self.robust_dark_profile(roi)
            if local_score is None or len(local_score) < 5:
                out["x2"] = float(body_x2 - OUTER_SNAP_INSET)
                out["side"] = "right"
                return out
            core_l = max(0, int(round(x1)) - search_l)
            core_r = min(len(local_score) - 1, int(round(x2)) - search_l)
            if core_r <= core_l:
                out["x2"] = float(body_x2 - OUTER_SNAP_INSET)
                out["side"] = "right"
                return out
            core = local_score[core_l:core_r + 1]
            core_max = float(np.max(core)) if len(core) > 0 else 0.0
            if core_max <= 1e-6:
                out["x2"] = float(body_x2 - OUTER_SNAP_INSET)
                out["side"] = "right"
                return out
            thr = INNER_THR_RATIO * core_max
            binary = (local_score >= thr).astype(np.uint8)
            binary = self.fill_small_gaps_1d(binary, MAX_GAP)
            runs = self.find_runs(binary)
            if not runs:
                out["x2"] = float(body_x2 - OUTER_SNAP_INSET)
                out["side"] = "right"
                return out
            chosen = self.choose_best_run_for_box(runs, core_l, core_r)
            if chosen is None:
                out["x2"] = float(body_x2 - OUTER_SNAP_INSET)
                out["side"] = "right"
                return out
            l, r = chosen
            edge = self.locate_inner_edge_1d(local_score, l, r, side="right")
            if edge is None:
                edge = l
            new_x1 = float(search_l + edge)
            new_x2 = float(body_x2 - OUTER_SNAP_INSET)
            if new_x2 - new_x1 < min_seg_len:
                new_x1 = float(search_l + l)
            if new_x2 <= new_x1 + 2:
                return box
            new_y1, new_y2 = self.refine_box_y(gray, new_x1, y1, new_x2, y2)
            out.update({"x1": new_x1, "x2": new_x2, "y1": new_y1, "y2": new_y2, "side": "right"})
            return out

    def post_smooth_side_labels(self, preds, body_x1, body_x2):
        out = []
        for p in preds:
            q = dict(p)
            q["side"] = self.classify_half_side(q, body_x1, body_x2)
            out.append(q)
        return out

    def clip_box(self, box, W, H):
        out = dict(box)
        out["x1"] = float(np.clip(out["x1"], 0, W - 2))
        out["x2"] = float(np.clip(out["x2"], out["x1"] + 1, W - 1))
        out["y1"] = float(np.clip(out["y1"], 0, H - 2))
        out["y2"] = float(np.clip(out["y2"], out["y1"] + 1, H - 1))
        return out

    def row_dark_profile_for_separator(self, gray, y1, y2, x1, x2):
        H, W = gray.shape[:2]
        x1 = max(0, min(W - 1, int(round(x1))))
        x2 = max(x1 + 1, min(W - 1, int(round(x2))))
        y1 = max(0, min(H - 1, int(round(y1))))
        y2 = max(y1 + 1, min(H - 1, int(round(y2))))
        roi = gray[y1:y2 + 1, x1:x2 + 1]
        if roi.size == 0:
            return None
        row_q = np.percentile(roi, 30, axis=1).astype(np.float32)
        row_score = 255.0 - row_q
        return smooth_1d(row_score, SEPARATOR_ROW_SMOOTH_SIGMA)

    def find_separator_y(self, gray, upper_box, lower_box, body_x1, body_x2):
        H, W = gray.shape[:2]
        upper_cy = 0.5 * (upper_box["y1"] + upper_box["y2"])
        lower_cy = 0.5 * (lower_box["y1"] + lower_box["y2"])
        search_y1 = int(round(upper_cy)) - SEPARATOR_SEARCH_PAD
        search_y2 = int(round(lower_cy)) + SEPARATOR_SEARCH_PAD
        search_y1 = max(0, min(H - 2, search_y1))
        search_y2 = max(search_y1 + 1, min(H - 1, search_y2))
        x1 = max(body_x1, min(upper_box["x1"], lower_box["x1"]))
        x2 = min(body_x2, max(upper_box["x2"], lower_box["x2"]))
        if x2 <= x1 + 2:
            x1, x2 = body_x1, body_x2
        row_score = self.row_dark_profile_for_separator(gray, search_y1, search_y2, x1, x2)
        if row_score is None or len(row_score) == 0:
            return int(round(0.5 * (upper_cy + lower_cy)))
        sep_local = int(np.argmin(row_score))
        return search_y1 + sep_local

    def resolve_vertical_overlaps(self, gray, preds, body_x1, body_x2, H, W):
        if len(preds) <= 1:
            return preds
        preds = self.sort_by_y([dict(p) for p in preds])
        for _ in range(OVERLAP_RESOLVE_PASSES):
            changed = False
            for i in range(len(preds) - 1):
                a, b = preds[i], preds[i + 1]
                if a["y2"] < b["y1"] - MIN_VERTICAL_GAP:
                    continue
                sep_y = self.find_separator_y(gray, a, b, body_x1, body_x2)
                new_a_y2 = min(a["y2"], sep_y - MIN_VERTICAL_GAP)
                new_b_y1 = max(b["y1"], sep_y + MIN_VERTICAL_GAP)
                if new_a_y2 - a["y1"] + 1 < MIN_BOX_H or b["y2"] - new_b_y1 + 1 < MIN_BOX_H:
                    hard_sep = int(round(0.5 * (a["y2"] + b["y1"])))
                    new_a_y2 = min(a["y2"], hard_sep - MIN_VERTICAL_GAP)
                    new_b_y1 = max(b["y1"], hard_sep + MIN_VERTICAL_GAP)
                if new_a_y2 - a["y1"] + 1 < MIN_BOX_H or b["y2"] - new_b_y1 + 1 < MIN_BOX_H:
                    ca = int(round(0.5 * (a["y1"] + a["y2"])))
                    cb = int(round(0.5 * (b["y1"] + b["y2"])))
                    mid = int(round(0.5 * (ca + cb)))
                    new_a_y2 = min(a["y2"], mid - MIN_VERTICAL_GAP)
                    new_b_y1 = max(b["y1"], mid + MIN_VERTICAL_GAP)
                if new_a_y2 - a["y1"] + 1 >= MIN_BOX_H and abs(new_a_y2 - a["y2"]) > 1e-6:
                    a["y2"] = float(new_a_y2)
                    changed = True
                if b["y2"] - new_b_y1 + 1 >= MIN_BOX_H and abs(new_b_y1 - b["y1"]) > 1e-6:
                    b["y1"] = float(new_b_y1)
                    changed = True
                preds[i] = self.clip_box(a, W, H)
                preds[i + 1] = self.clip_box(b, W, H)
            preds = self.sort_by_y(preds)
            if not changed:
                break
        return preds

    def contour_from_mask(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        return None if cv2.contourArea(cnt) < 5 else cnt

    def crop_box_with_pad(self, gray, box, W, H, pad_x=2, pad_y=2):
        x1 = max(0, int(math.floor(box["x1"])) - pad_x)
        y1 = max(0, int(math.floor(box["y1"])) - pad_y)
        x2 = min(W - 1, int(math.ceil(box["x2"])) + pad_x)
        y2 = min(H - 1, int(math.ceil(box["y2"])) + pad_y)
        roi = gray[y1:y2 + 1, x1:x2 + 1]
        return roi, x1, y1, x2, y2

    def get_center_band_bounds(self, h):
        band_h = max(MIN_CENTER_BAND_H, int(round(h * CENTER_BAND_RATIO)))
        band_h = min(band_h, h)
        cy = h // 2
        y1 = max(0, cy - band_h // 2)
        y2 = min(h - 1, y1 + band_h - 1)
        return y1, y2

    def estimate_line_band_rows(self, roi_gray):
        h, w = roi_gray.shape[:2]
        if h < 3 or w < 5:
            return None
        band_y1, band_y2 = self.get_center_band_bounds(h)
        band = roi_gray[band_y1:band_y2 + 1, :]
        if band.size == 0:
            return None
        row_q = np.percentile(band, ROW_PERCENTILE, axis=1).astype(np.float32)
        row_score = 255.0 - row_q
        row_score = smooth_1d(row_score, ROW_SMOOTH_SIGMA)
        peak = float(np.max(row_score))
        if peak <= 1e-6:
            return None
        thr = max(ROW_THR_RATIO * peak, np.percentile(row_score, 55))
        row_mask = (row_score >= thr).astype(np.uint8)
        row_mask = self.fill_small_gaps_1d(row_mask, max_gap=2)
        runs = self.find_runs(row_mask)
        if not runs:
            fallback_h = max(3, int(round((band_y2 - band_y1 + 1) * FALLBACK_HEIGHT_RATIO)))
            cy = 0.5 * (band_y1 + band_y2)
            return max(0, int(round(cy - fallback_h / 2))), min(h - 1, int(round(cy + fallback_h / 2)))
        center_y_local = 0.5 * (band_y2 - band_y1)
        chosen = None
        for a, b in runs:
            if a <= center_y_local <= b:
                chosen = (a, b)
                break
        if chosen is None:
            chosen = max(runs, key=lambda t: t[1] - t[0] + 1)
        y1 = band_y1 + chosen[0]
        y2 = band_y1 + chosen[1]
        if y2 <= y1:
            fallback_h = max(3, int(round((band_y2 - band_y1 + 1) * FALLBACK_HEIGHT_RATIO)))
            cy = 0.5 * (band_y1 + band_y2)
            y1 = max(0, int(round(cy - fallback_h / 2)))
            y2 = min(h - 1, int(round(cy + fallback_h / 2)))
        return y1, y2

    def estimate_centerline_in_band(self, roi_gray, y1, y2):
        h, w = roi_gray.shape[:2]
        y1 = max(0, min(h - 1, int(round(y1))))
        y2 = max(y1 + 1, min(h - 1, int(round(y2))))
        band = roi_gray[y1:y2 + 1, :]
        if band.size == 0:
            return None
        pts = []
        for x in range(w):
            col = band[:, x].astype(np.float32)
            yy = int(np.argmin(col))
            pts.append([x, y1 + yy])
        pts = np.array(pts, dtype=np.float32)
        if len(pts) < 5:
            return None
        line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = [float(v) for v in line.reshape(-1)]
        norm = math.hypot(vx, vy)
        if norm <= 1e-6:
            return None
        vx /= norm
        vy /= norm
        angle_deg = normalize_angle_180(math.degrees(math.atan2(vy, vx)))
        if abs(vx) < 1e-6:
            cy = 0.5 * (y1 + y2)
            p1 = np.array([0.0, cy], dtype=np.float32)
            p2 = np.array([float(w - 1), cy], dtype=np.float32)
        else:
            k = vy / vx
            p1 = np.array([0.0, y0 + k * (0.0 - x0)], dtype=np.float32)
            p2 = np.array([float(w - 1), y0 + k * ((w - 1) - x0)], dtype=np.float32)
        center = 0.5 * (p1 + p2)
        length_px = float(np.linalg.norm(p2 - p1))
        width_px = float(y2 - y1 + 1)
        nx, ny = -vy, vx
        w1 = center - 0.5 * width_px * np.array([nx, ny], dtype=np.float32)
        w2 = center + 0.5 * width_px * np.array([nx, ny], dtype=np.float32)
        return {
            "angle_deg": angle_deg,
            "length_px": length_px,
            "width_px": width_px,
            "center": center,
            "line_p1": p1,
            "line_p2": p2,
            "width_p1": w1,
            "width_p2": w2,
        }

    def detect_line_in_box_centerband(self, gray, box, body_x1, body_x2, W, H):
        side = box.get("side", self.classify_half_side(box, body_x1, body_x2))
        roi, gx1, gy1, gx2, gy2 = self.crop_box_with_pad(gray, box, W, H, pad_x=ROI_PAD_X, pad_y=ROI_PAD_Y)
        if roi.size == 0:
            return None
        row_bounds = self.estimate_line_band_rows(roi)
        if row_bounds is None:
            return None
        y1, y2 = row_bounds
        mask = np.zeros_like(roi, dtype=np.uint8)
        mask[y1:y2 + 1, :] = 255
        if np.count_nonzero(mask) < MIN_MASK_AREA:
            return None
        contour = self.contour_from_mask(mask)
        if contour is None:
            return None
        measure = self.estimate_centerline_in_band(roi, y1, y2)
        if measure is None:
            return None
        contour_global = contour.copy().astype(np.int32)
        contour_global[:, 0, 0] += gx1
        contour_global[:, 0, 1] += gy1

        def to_global(pt):
            return np.array([pt[0] + gx1, pt[1] + gy1], dtype=np.float32)

        return {
            "side": side,
            "roi_box": (gx1, gy1, gx2, gy2),
            "mask": mask,
            "contour_global": contour_global,
            "angle_deg": measure["angle_deg"],
            "length_px": measure["length_px"],
            "width_px": measure["width_px"],
            "center_global": to_global(measure["center"]),
            "line_p1_global": to_global(measure["line_p1"]),
            "line_p2_global": to_global(measure["line_p2"]),
            "width_p1_global": to_global(measure["width_p1"]),
            "width_p2_global": to_global(measure["width_p2"]),
        }

    def draw_measurement(self, vis, det, idx_text=None, parallel_text=None, mm_per_px: Optional[float] = None):
        if DRAW_CONTOUR:
            cv2.drawContours(vis, [det["contour_global"]], -1, (0, 255, 255), 2)
        lp1 = tuple(np.round(det["line_p1_global"]).astype(int))
        lp2 = tuple(np.round(det["line_p2_global"]).astype(int))
        wp1 = tuple(np.round(det["width_p1_global"]).astype(int))
        wp2 = tuple(np.round(det["width_p2_global"]).astype(int))
        center = tuple(np.round(det["center_global"]).astype(int))
        if DRAW_CENTERLINE:
            cv2.line(vis, lp1, lp2, (255, 255, 0), 2)
            cv2.circle(vis, lp1, 3, (255, 255, 0), -1)
            cv2.circle(vis, lp2, 3, (255, 255, 0), -1)
        if DRAW_WIDTH_LINE:
            cv2.line(vis, wp1, wp2, (255, 0, 255), 2)

        text_y = max(15, center[1] - 8)

        if DRAW_LENGTH_TEXT:
            txt = f"L={det['length_px']:.1f}px"
            if mm_per_px is not None:
                txt += f" / {det['length_px'] * mm_per_px:.4f}mm"
            if idx_text is not None:
                txt = f"{idx_text} {txt}"
            cv2.putText(vis, txt, (center[0] + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if DRAW_WIDTH_TEXT:
            txt = f"W={det['width_px']:.1f}px"
            if mm_per_px is not None:
                txt += f" / {det['width_px'] * mm_per_px:.4f}mm"
            cv2.putText(vis, txt, (center[0] + 5, text_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        if DRAW_PARALLEL_TEXT and parallel_text is not None:
            cv2.putText(vis, parallel_text, (center[0] + 5, text_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

    def measure(self, roi_bgr: np.ndarray, out_dir: str, roi_name: str,
                mm_per_px: Optional[float] = None) -> Tuple[List[LineMetric], str, List[dict]]:
        ensure_dir(out_dir)
        img = roi_bgr.copy()
        H, W = img.shape[:2]
        gray = enhance_gray(img)
        body_x1, body_x2 = self.estimate_resistor_body_x(gray)

        results = self.model.predict(
            source=img,
            imgsz=LINE_IMGSZ,
            conf=LINE_CONF,
            iou=LINE_IOU,
            device=DEVICE,
            verbose=False,
        )
        r = results[0]
        boxes = r.boxes
        preds = []
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                preds.append({
                    "cls": int(b.cls.item()),
                    "conf": float(b.conf.item()),
                    "x1": float(b.xyxy[0][0].item()),
                    "y1": float(b.xyxy[0][1].item()),
                    "x2": float(b.xyxy[0][2].item()),
                    "y2": float(b.xyxy[0][3].item()),
                })
        preds = self.deduplicate_boxes(preds, W, H)
        preds = self.sort_by_y(preds)
        preds = self.post_smooth_side_labels(preds, body_x1, body_x2)

        refined_preds = []
        for p in preds:
            refined = self.refine_half_box(gray, p, body_x1, body_x2)
            refined = self.clip_box(refined, W, H)
            refined_preds.append(refined)

        preds = self.resolve_vertical_overlaps(gray, refined_preds, body_x1, body_x2, H, W)
        preds = self.sort_by_y(preds)

        measured = []
        for p in preds:
            det = self.detect_line_in_box_centerband(gray, p, body_x1, body_x2, W, H)
            if det is None:
                continue
            det["cls"] = p["cls"]
            det["conf"] = p["conf"]
            det["refined_box"] = p
            measured.append(det)

        measured = sorted(measured, key=lambda d: d["center_global"][1])

        line_metrics = []
        for i, det in enumerate(measured):
            if i < len(measured) - 1:
                pdiff = angle_diff_deg(det["angle_deg"], measured[i + 1]["angle_deg"])
                is_parallel = judge_parallel_yes_no(pdiff)
                # 修正 gap 方向：下一条上边界 - 当前条下边界
                next_gap = float(measured[i + 1]["refined_box"]["y1"] - det["refined_box"]["y2"])
            else:
                pdiff = None
                is_parallel = None
                next_gap = None

            p = det["refined_box"]

            length_real_mm = det["length_px"] * mm_per_px if mm_per_px is not None else None
            width_real_mm = det["width_px"] * mm_per_px if mm_per_px is not None else None
            next_gap_real_mm = next_gap * mm_per_px if (mm_per_px is not None and next_gap is not None) else None

            line_metrics.append(LineMetric(
                line_index=i,
                side=det["side"],
                cls=det["cls"],
                conf=det["conf"],
                box_x1=p["x1"],
                box_y1=p["y1"],
                box_x2=p["x2"],
                box_y2=p["y2"],
                length_px=det["length_px"],
                width_px=det["width_px"],
                length_real_mm=length_real_mm,
                width_real_mm=width_real_mm,
                angle_deg=det["angle_deg"],
                parallel_diff_next_deg=pdiff,
                is_parallel_next=is_parallel,
                next_gap_y=next_gap,
                next_gap_real_mm=next_gap_real_mm,
                center_x=float(det["center_global"][0]),
                center_y=float(det["center_global"][1]),
            ))

        vis = img.copy()
        if DRAW_BODY_EDGE:
            cv2.line(vis, (int(body_x1), 0), (int(body_x1), H - 1), (255, 0, 0), 1)
            cv2.line(vis, (int(body_x2), 0), (int(body_x2), H - 1), (255, 0, 0), 1)

        for p in preds:
            x1i, y1i, x2i, y2i = map(int, [p["x1"], p["y1"], p["x2"], p["y2"]])
            cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            if DRAW_SIDE_TEXT:
                side = p.get("side", self.classify_half_side(p, body_x1, body_x2))
                cv2.putText(vis, f"{side[0].upper()}:{p['conf']:.2f}",
                            (x1i, max(12, y1i - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        for i, det in enumerate(measured):
            ptxt = None
            if i < len(line_metrics) and line_metrics[i].parallel_diff_next_deg is not None:
                ptxt = f"Par={line_metrics[i].parallel_diff_next_deg:.2f}deg ({line_metrics[i].is_parallel_next})"
            self.draw_measurement(vis, det, idx_text=f"#{i}", parallel_text=ptxt, mm_per_px=mm_per_px)

        vis_path = os.path.join(out_dir, f"{os.path.splitext(roi_name)[0]}_vis.png")
        cv2.imwrite(vis_path, vis)
        return line_metrics, vis_path, preds


class SpecJudge:
    def __init__(self, spec_csv: str):
        self.df = pd.read_csv(spec_csv) if os.path.exists(spec_csv) else None

    def judge_roi(self, roi_id: int, line_metrics: List[LineMetric]) -> Tuple[str, str]:
        if self.df is None or self.df.empty:
            return "unknown", "未加载标准规格表"
        row = self.df[self.df["resistor_id"] == roi_id]
        if row.empty:
            return "unknown", f"未找到 resistor_id={roi_id} 的规格"
        spec = row.iloc[0]
        fails = []

        # 这里仍沿用像素规格判定；如果你后面规格表要改成 mm，再把这里改成真实尺寸比较
        for lm in line_metrics:
            if "length_min" in spec and pd.notna(spec["length_min"]) and lm.length_px < float(spec["length_min"]):
                fails.append(f"line{lm.line_index}: length<{spec['length_min']}")
            if "length_max" in spec and pd.notna(spec["length_max"]) and lm.length_px > float(spec["length_max"]):
                fails.append(f"line{lm.line_index}: length>{spec['length_max']}")
            if "width_min" in spec and pd.notna(spec["width_min"]) and lm.width_px < float(spec["width_min"]):
                fails.append(f"line{lm.line_index}: width<{spec['width_min']}")
            if "width_max" in spec and pd.notna(spec["width_max"]) and lm.width_px > float(spec["width_max"]):
                fails.append(f"line{lm.line_index}: width>{spec['width_max']}")
            if lm.next_gap_y is not None:
                if "gap_min" in spec and pd.notna(spec["gap_min"]) and lm.next_gap_y < float(spec["gap_min"]):
                    fails.append(f"line{lm.line_index}: gap<{spec['gap_min']}")
                if "gap_max" in spec and pd.notna(spec["gap_max"]) and lm.next_gap_y > float(spec["gap_max"]):
                    fails.append(f"line{lm.line_index}: gap>{spec['gap_max']}")
            if str(spec.get("parallel_required", "")).lower() == "yes" and lm.is_parallel_next not in [None, "yes"]:
                fails.append(f"line{lm.line_index}: not parallel")

        return ("pass", "") if not fails else ("fail", "; ".join(fails))


class ResistorAppPipeline:
    def __init__(self):
        ensure_dir(OUTPUT_ROOT)
        self.roi_detector = RoiDetector(WEIGHTS_ROI)
        self.sr_engine = SuperResolutionEngine(REAL_ESRGAN_MODEL_PATH, ENABLE_SR, SR_SCALE)
        self.line_engine = LineMeasurementEngine(WEIGHTS_LINE)
        self.spec_judge = SpecJudge(STANDARD_SPEC_CSV)

    def run_one_image(self, image_path: str,
                      manual_roi_id: Optional[int] = None,
                      mm_per_px: Optional[float] = None) -> Dict:
        image_name = os.path.basename(image_path)
        image_stem = os.path.splitext(image_name)[0]
        out_dir = os.path.join(OUTPUT_ROOT, image_stem)
        roi_dir = os.path.join(out_dir, "rois")
        sr_dir = os.path.join(out_dir, "sr")
        vis_dir = os.path.join(out_dir, "vis")
        ensure_dir(out_dir)
        ensure_dir(roi_dir)
        ensure_dir(sr_dir)
        ensure_dir(vis_dir)

        big = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if big is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        rois = self.roi_detector.detect(big, image_name)
        if manual_roi_id is not None:
            rois = [r for r in rois if r.roi_id == manual_roi_id]

        all_roi_results: List[RoiResult] = []
        overlay = big.copy()

        for roi_meta in rois:
            roi = big[roi_meta.y1:roi_meta.y2, roi_meta.x1:roi_meta.x2].copy()
            roi_path = os.path.join(roi_dir, roi_meta.roi_name)
            cv2.imwrite(roi_path, roi)

            sr_img = self.sr_engine.run(roi)
            sr_path = os.path.join(sr_dir, roi_meta.roi_name)
            cv2.imwrite(sr_path, sr_img)

            line_metrics, roi_vis_path, _ = self.line_engine.measure(
                sr_img, vis_dir, roi_meta.roi_name, mm_per_px=mm_per_px
            )
            status, reason = self.spec_judge.judge_roi(roi_meta.roi_id, line_metrics)

            all_roi_results.append(RoiResult(
                roi_meta=roi_meta,
                num_lines=len(line_metrics),
                status=status,
                fail_reasons=reason,
                line_metrics=line_metrics,
                vis_path=roi_vis_path,
                sr_path=sr_path,
            ))

            color = COLOR_PASS if status in ["pass", "unknown"] else COLOR_FAIL
            cv2.rectangle(overlay, (roi_meta.x1, roi_meta.y1), (roi_meta.x2, roi_meta.y2), color, 3)
            # label = f"ROI{roi_meta.roi_id:03d} {status}"
            # cv2.putText(overlay, label, (roi_meta.x1, max(20, roi_meta.y1 - 8)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # cv2.putText(overlay, f"lines={len(line_metrics)}", (roi_meta.x1, min(big.shape[0] - 10, roi_meta.y2 + 22)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ROI_ID, 2)

        overlay_path = os.path.join(out_dir, f"{image_stem}_overlay.png")
        cv2.imwrite(overlay_path, overlay)

        roi_summary_rows = []
        line_detail_rows = []
        for rr in all_roi_results:
            roi_summary_rows.append({
                "image": image_name,
                "roi_id": rr.roi_meta.roi_id,
                "roi_name": rr.roi_meta.roi_name,
                "x1": rr.roi_meta.x1,
                "y1": rr.roi_meta.y1,
                "x2": rr.roi_meta.x2,
                "y2": rr.roi_meta.y2,
                "conf": rr.roi_meta.conf,
                "num_lines": rr.num_lines,
                "status": rr.status,
                "fail_reasons": rr.fail_reasons,
                "roi_vis_path": rr.vis_path,
                "roi_sr_path": rr.sr_path,
                "mm_per_px": mm_per_px,
            })
            for lm in rr.line_metrics:
                line_detail_rows.append({
                    "image": image_name,
                    "roi_id": rr.roi_meta.roi_id,
                    **asdict(lm),
                    "roi_status": rr.status,
                    "roi_fail_reasons": rr.fail_reasons,
                    "mm_per_px": mm_per_px,
                })

        summary_csv = os.path.join(out_dir, "roi_summary.csv")
        detail_csv = os.path.join(out_dir, "line_details.csv")
        pd.DataFrame(roi_summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(line_detail_rows).to_csv(detail_csv, index=False, encoding="utf-8-sig")

        return {
            "image_path": image_path,
            "overlay_path": overlay_path,
            "summary_csv": summary_csv,
            "detail_csv": detail_csv,
            "out_dir": out_dir,
            "roi_results": all_roi_results,
            "mm_per_px": mm_per_px,
        }


PIPELINE = None


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = ResistorAppPipeline()
    return PIPELINE


def load_first_demo_image():
    paths = []
    for ext in IMG_EXTS:
        paths.extend(sorted(glob.glob(os.path.join(INPUT_DIR, f"*{ext}"))))
    if not paths:
        return None
    img = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    if img is None:
        return None
    return bgr_to_rgb(img)


def build_demo_choices():
    choices = []
    for ext in IMG_EXTS:
        choices.extend(sorted(glob.glob(os.path.join(INPUT_DIR, f"*{ext}"))))
    return choices


def choose_demo_image(path):
    if not path:
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return bgr_to_rgb(img)


# ============================
# 标定相关交互函数
# ============================

def use_current_input_as_calibration(input_image, calibration_state):
    if input_image is None:
        msg = "当前输入图为空，无法作为标定图。"
        return None, calibration_state, msg
    calibration_state = init_calibration_state()
    preview = draw_calibration_overlay(input_image, calibration_state)
    msg = (
        "已加载当前输入图作为标定图。\n"
        "请在图上依次点击：已知 5 mm 参考物的左边界、右边界。"
    )
    return preview, calibration_state, msg


def load_demo_to_calibration(path, calibration_state):
    img = choose_demo_image(path)
    if img is None:
        return None, calibration_state, "示例图读取失败。"
    calibration_state = init_calibration_state()
    preview = draw_calibration_overlay(img, calibration_state)
    msg = (
        f"已加载示例图到标定区：{path}\n"
        "请在图上依次点击：已知 5 mm 参考物的左边界、右边界。"
    )
    return preview, calibration_state, msg


def upload_calibration_image(calib_img):
    state = init_calibration_state()
    if calib_img is None:
        return None, state, "请先上传标定图。"
    preview = draw_calibration_overlay(calib_img, state)
    msg = "标定图已加载。请依次点击已知 5 mm 参考物的左边界、右边界。"
    return preview, state, msg


def reset_calibration(calib_img):
    state = init_calibration_state()
    if calib_img is None:
        return None, state, "比例尺已清除。请重新加载标定图。", "未标定"
    preview = draw_calibration_overlay(calib_img, state)
    return preview, state, "比例尺已清除，请重新点击两点完成标定。", "未标定"


def calibration_click(calib_img, calibration_state, evt: gr.SelectData):
    if calib_img is None:
        return None, calibration_state, "请先加载标定图。", "未标定"

    if calibration_state is None or not isinstance(calibration_state, dict):
        calibration_state = init_calibration_state()

    x, y = evt.index
    pts = calibration_state.get("points", [])

    # 最多保留两点；如果已经有两点，再点一次就重新开始
    if len(pts) >= 2:
        pts = []

    pts.append((int(x), int(y)))
    calibration_state["points"] = pts

    if len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        pixel_length = float(math.hypot(x2 - x1, y2 - y1))
        if pixel_length <= 1e-6:
            calibration_state["pixel_length"] = None
            calibration_state["mm_per_px"] = None
            calibration_state["is_calibrated"] = False
            preview = draw_calibration_overlay(calib_img, calibration_state)
            return preview, calibration_state, "两点距离过小，请重新点击。", "未标定"

        mm_per_px = KNOWN_CALIBRATION_LENGTH_MM / pixel_length
        calibration_state["pixel_length"] = pixel_length
        calibration_state["mm_per_px"] = mm_per_px
        calibration_state["is_calibrated"] = True
        calibration_state["known_length_mm"] = KNOWN_CALIBRATION_LENGTH_MM

        preview = draw_calibration_overlay(calib_img, calibration_state)
        status_text = (
            f"已标定：5.0 mm / {pixel_length:.2f} px = {mm_per_px:.6f} mm/px"
        )
        msg = (
            f"标定完成。\n"
            f"已知长度: {KNOWN_CALIBRATION_LENGTH_MM:.2f} mm\n"
            f"像素长度: {pixel_length:.2f} px\n"
            f"比例尺: {mm_per_px:.6f} mm/px\n"
            f"后续测量将沿用此比例尺，直到你点击“重新测量比例尺 / 清除比例尺”。"
        )
        return preview, calibration_state, msg, status_text

    preview = draw_calibration_overlay(calib_img, calibration_state)
    status_text = f"已选 {len(pts)} / 2 点"
    msg = f"已记录第 {len(pts)} 个点，请继续点击第 {len(pts)+1} 个点。"
    return preview, calibration_state, msg, status_text


# ============================
# 推理函数
# ============================

def run_web_inference(input_image, manual_roi_id_text, calibration_state):
    try:
        pipeline = get_pipeline()
        if input_image is None:
            raise ValueError("请先上传一张图像")

        if calibration_state is None or not isinstance(calibration_state, dict):
            calibration_state = init_calibration_state()

        mm_per_px = calibration_state.get("mm_per_px", None)
        if mm_per_px is None:
            raise ValueError("当前尚未完成比例尺标定。请先在标定区点击两点完成 5 mm 标定。")

        manual_roi_id = None
        if manual_roi_id_text is not None and str(manual_roi_id_text).strip() != "":
            manual_roi_id = int(str(manual_roi_id_text).strip())

        bgr = np_to_bgr(input_image)
        ensure_dir(OUTPUT_ROOT)
        temp_input_path = os.path.join(OUTPUT_ROOT, "_web_input.png")
        cv2.imwrite(temp_input_path, bgr)

        result = pipeline.run_one_image(temp_input_path, manual_roi_id=manual_roi_id, mm_per_px=mm_per_px)
        overlay = cv2.imread(result["overlay_path"], cv2.IMREAD_COLOR)
        overlay_rgb = bgr_to_rgb(overlay)

        summary_df = pd.read_csv(result["summary_csv"]) if os.path.exists(result["summary_csv"]) else pd.DataFrame()
        detail_df = pd.read_csv(result["detail_csv"]) if os.path.exists(result["detail_csv"]) else pd.DataFrame()

        text = []
        text.append(f"输出目录: {result['out_dir']}")
        text.append(f"ROI 汇总表: {result['summary_csv']}")
        text.append(f"线条明细表: {result['detail_csv']}")
        text.append(f"检测到 ROI 数量: {len(result['roi_results'])}")
        text.append(f"当前比例尺: {mm_per_px:.6f} mm/px")
        return overlay_rgb, summary_df, detail_df, "\n".join(text), result["summary_csv"], result["detail_csv"]
    except Exception as e:
        err = traceback.format_exc()
        return None, pd.DataFrame(), pd.DataFrame(), f"运行失败:\n{e}\n\n{err}", None, None


def make_ui():
    demo_paths = build_demo_choices()

    with gr.Blocks(title="多电阻图像自动测量系统") as demo:
        calibration_state = gr.State(init_calibration_state())

        gr.Markdown("# 多电阻图像自动测量系统（网页版）")
        gr.Markdown(
            "使用流程：**先在顶部完成比例尺标定**（加载含已知 5 mm 参考物的图，并点击左右边界），"
            "再在下方上传待检测大图并执行测量。标定完成后，系统会持续沿用该比例尺，直到你手动清除并重新标定。"
        )

        # =========================================================
        # 1. 顶部：比例尺标定区
        # =========================================================
        gr.Markdown("## ① 比例尺标定区（请先完成）")

        with gr.Row():
            with gr.Column(scale=1):
                calibration_image = gr.Image(
                    label="上传标定图（可单独上传）",
                    type="numpy"
                )

                demo_dropdown_for_calib = gr.Dropdown(
                    choices=demo_paths,
                    label="或选择 INPUT_DIR 中的示例图作为标定图",
                    value=demo_paths[0] if demo_paths else None
                )

                with gr.Row():
                    use_demo_btn = gr.Button("使用当前示例图作为标定图")
                    reset_calib_btn = gr.Button("重新测量比例尺 / 清除比例尺", variant="stop")

            with gr.Column(scale=1):
                calibration_preview = gr.Image(
                    label="标定预览图（点击 5 mm 参考物左右边界）",
                    type="numpy",
                    interactive=True
                )
                calibration_status = gr.Textbox(
                    label="当前比例尺状态",
                    value="未标定"
                )
                calibration_log = gr.Textbox(
                    label="标定日志",
                    lines=8
                )

        # =========================================================
        # 2. 中间：待检测图输入区
        # =========================================================
        gr.Markdown("## ② 待检测图输入与结果显示")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="上传输入图像（待检测大图）",
                    type="numpy"
                )

                demo_dropdown = gr.Dropdown(
                    choices=demo_paths,
                    label="或选择 INPUT_DIR 中的示例图作为待检测图",
                    value=demo_paths[0] if demo_paths else None
                )

                use_input_btn = gr.Button("使用当前待检测图作为标定图")

                manual_roi_id = gr.Textbox(
                    label="手动 ROI 编号（可空）",
                    placeholder="例如 3"
                )

                run_btn = gr.Button("开始测量", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(label="结果叠加图")

        # =========================================================
        # 3. 底部：表格、日志、下载
        # =========================================================
        gr.Markdown("## ③ 测量结果与导出")

        with gr.Row():
            summary_table = gr.Dataframe(label="ROI 汇总表")
            detail_table = gr.Dataframe(label="线条明细表")

        log_text = gr.Textbox(label="运行日志", lines=10)

        with gr.Row():
            summary_file = gr.File(label="下载 ROI 汇总 CSV")
            detail_file = gr.File(label="下载线条明细 CSV")

        # ============================
        # 事件绑定
        # ============================

        # 选择示例图 -> 加载到待检测输入区
        demo_dropdown.change(
            fn=choose_demo_image,
            inputs=demo_dropdown,
            outputs=input_image
        )

        # 选择示例图 -> 加载到标定区
        demo_dropdown_for_calib.change(
            fn=choose_demo_image,
            inputs=demo_dropdown_for_calib,
            outputs=calibration_image
        )

        # 单独上传标定图
        calibration_image.change(
            fn=upload_calibration_image,
            inputs=calibration_image,
            outputs=[calibration_preview, calibration_state, calibration_log]
        )

        # 使用当前待检测图作为标定图
        use_input_btn.click(
            fn=use_current_input_as_calibration,
            inputs=[input_image, calibration_state],
            outputs=[calibration_preview, calibration_state, calibration_log]
        )

        # 使用当前示例图作为标定图
        use_demo_btn.click(
            fn=load_demo_to_calibration,
            inputs=[demo_dropdown_for_calib, calibration_state],
            outputs=[calibration_preview, calibration_state, calibration_log]
        )

        # 点击标定图，记录两点
        calibration_preview.select(
            fn=calibration_click,
            inputs=[calibration_preview, calibration_state],
            outputs=[calibration_preview, calibration_state, calibration_log, calibration_status]
        )

        # 清除比例尺
        reset_calib_btn.click(
            fn=reset_calibration,
            inputs=[calibration_preview],
            outputs=[calibration_preview, calibration_state, calibration_log, calibration_status]
        )

        # 主检测按钮
        run_btn.click(
            fn=run_web_inference,
            inputs=[input_image, manual_roi_id, calibration_state],
            outputs=[output_image, summary_table, detail_table, log_text, summary_file, detail_file],
        )

        # 页面初始化
        if demo_paths:
            demo.load(
                fn=lambda: choose_demo_image(demo_paths[0]),
                inputs=None,
                outputs=input_image
            )
            demo.load(
                fn=lambda: choose_demo_image(demo_paths[0]),
                inputs=None,
                outputs=calibration_image
            )

    return demo


if __name__ == "__main__":
    ensure_dir(OUTPUT_ROOT)
    ui = make_ui()
    ui.launch(
        server_name=WEB_HOST,
        server_port=WEB_PORT,
        share=WEB_SHARE,
        allowed_paths=[OUTPUT_ROOT]
    )