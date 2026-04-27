import os
import shutil
from glob import glob
import cv2
import numpy as np
import torch

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


# ===================== 你只改这里 =====================
SRC_ROOT = "/root/ultralytics-main/roi_line_yolo_dataset"   # 你的原始YOLO数据集根目录（含 images/labels）
DST_ROOT = "/root/ultralytics-main/roi_line_yolo_dataset_mix"  # 输出混合域数据集根目录

MODEL_PATH = "/root/Real-ESRGAN/weights/RealESRGAN_x4plus.pth"  # ESRGAN权重
SR_SCALE = 2                 # SR倍数：2 或 4（建议先2）
SR_RESIZE_BACK = False       # True: SR后resize回原尺寸；False: 保持放大尺寸

MAKE_RAW = True              # 是否保留原图副本
MAKE_SR  = True              # 是否生成SR副本
MAKE_ENH = True              # 是否生成增强副本（建议对 SR 或 raw 做增强）

ENH_BASE = "sr"              # "sr" 或 "raw"：增强图是基于SR还是原图
# =====================================================


def ensure_dir(p): os.makedirs(p, exist_ok=True)


# ---------- 你的增强函数（输出改成3通道，便于YOLO直接训练） ----------
def enhance_for_horizontal_scratches_to_bgr(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    gray_dn = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=7)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_ce = clahe.apply(gray_dn)

    k_len = 35
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    tophat = cv2.morphologyEx(gray_ce, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray_ce, cv2.MORPH_BLACKHAT, kernel)
    resp = cv2.addWeighted(tophat, 1.0, blackhat, 1.0, 0)

    blur = cv2.GaussianBlur(gray_ce, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray_ce, 1.4, blur, -0.4, 0)

    enhanced = cv2.addWeighted(sharp, 0.85, resp, 0.75, 0)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    # 转为 3 通道
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ---------- 构建 Real-ESRGAN upsampler（一次构建，多次复用） ----------
def build_upsampler(model_path: str, scale: int):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SR model not found: {model_path}")

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23,
        num_grow_ch=32, scale=scale
    )
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,            # 显存不够可改 256/512
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()
    )
    return upsampler


def sr_image(upsampler: RealESRGANer, bgr: np.ndarray, outscale: int, resize_back_hw=None):
    out, _ = upsampler.enhance(bgr, outscale=outscale)
    if resize_back_hw is not None:
        h, w = resize_back_hw
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    return out


def copy_label(src_lab: str, dst_lab: str):
    if not os.path.exists(src_lab):
        # 允许没有label（背景图）
        return
    ensure_dir(os.path.dirname(dst_lab))
    shutil.copy2(src_lab, dst_lab)


def process_split(split: str, upsampler: RealESRGANer | None):
    src_img_dir = os.path.join(SRC_ROOT, "images", split)
    src_lab_dir = os.path.join(SRC_ROOT, "labels", split)

    if not os.path.isdir(src_img_dir):
        print(f"[Skip] split not found: {src_img_dir}")
        return

    dst_img_dir = os.path.join(DST_ROOT, "images", split)
    dst_lab_dir = os.path.join(DST_ROOT, "labels", split)
    ensure_dir(dst_img_dir)
    ensure_dir(dst_lab_dir)

    img_paths = sorted(glob(os.path.join(src_img_dir, "*.*")))
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    for p in img_paths:
        ext = os.path.splitext(p)[1].lower()
        if ext not in exts:
            continue

        name = os.path.splitext(os.path.basename(p))[0]
        lab_path = os.path.join(src_lab_dir, f"{name}.txt")

        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[Warn] cannot read: {p}")
            continue
        H, W = bgr.shape[:2]

        # 1) raw
        if MAKE_RAW:
            out_name = f"{name}_raw.png"
            out_img = os.path.join(dst_img_dir, out_name)
            cv2.imwrite(out_img, bgr)
            copy_label(lab_path, os.path.join(dst_lab_dir, f"{name}_raw.txt"))

        # 2) sr
        bgr_sr = None
        if MAKE_SR:
            if upsampler is None:
                raise RuntimeError("upsampler is None but MAKE_SR=True")

            resize_back = (H, W) if SR_RESIZE_BACK else None
            bgr_sr = sr_image(upsampler, bgr, outscale=SR_SCALE, resize_back_hw=resize_back)

            out_name = f"{name}_sr{SR_SCALE}.png"
            out_img = os.path.join(dst_img_dir, out_name)
            cv2.imwrite(out_img, bgr_sr)
            copy_label(lab_path, os.path.join(dst_lab_dir, f"{name}_sr{SR_SCALE}.txt"))

        # 3) enh（建议对SR做增强）
        if MAKE_ENH:
            base = None
            if ENH_BASE.lower() == "sr":
                if bgr_sr is None:
                    # 没开SR就退回raw
                    base = bgr
                else:
                    base = bgr_sr
            else:
                base = bgr

            enh = enhance_for_horizontal_scratches_to_bgr(base)

            suffix = "enh_fromSR" if (ENH_BASE.lower() == "sr" and bgr_sr is not None) else "enh"
            out_name = f"{name}_{suffix}.png"
            out_img = os.path.join(dst_img_dir, out_name)
            cv2.imwrite(out_img, enh)

            copy_label(lab_path, os.path.join(dst_lab_dir, f"{name}_{suffix}.txt"))

    print(f"[OK] split done: {split}")


def main():
    ensure_dir(DST_ROOT)
    upsampler = None
    if MAKE_SR:
        upsampler = build_upsampler(MODEL_PATH, scale=4)  # x4plus 权重配置固定是 scale=4
        # 注意：即使 SR_SCALE=2，这里仍用 x4plus 的 RRDBNet(scale=4) 构建是官方常用写法，
        # outscale 控制最终放大倍率。

    for split in ["train", "val", "test"]:
        process_split(split, upsampler)

    print("\n[DONE] Mixed-domain YOLO dataset created at:")
    print("  ", DST_ROOT)
    print("\nNotes:")
    print(" - YOLO labels are normalized, so SR resizing (uniform scaling) does NOT require label changes.")
    print(" - If you ever crop/letterbox-save images, then labels must be updated accordingly.")


if __name__ == "__main__":
    main()