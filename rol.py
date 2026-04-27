import os
import csv
import glob
import random
import shutil
import math
import cv2
from ultralytics import YOLO

# ======================= 配置区域 =======================
# 1) 电阻ROI检测模型（用于裁剪）
WEIGHTS_ROI = "/root/ultralytics-main/best.pt"
IMG_DIR = "/root/ultralytics-main/dataset/image"
ROI_DIR = "/root/ultralytics-main/rois_test"  # 输出roi图片 + metadata.csv

CONF = 0.25
IOU = 0.6
PAD_RATIO = 0.10  # 外扩比例：0.08~0.15

# 2) 大图线条 YOLO 标签目录（与你的大图同名 txt）
BIG_LINE_LABEL_DIR = "/root/ultralytics-main/dataset/labels"

# 3) ROI 标签输出目录（会在 ROI_DIR 下生成 labels）
ROI_LABEL_DIR = os.path.join(ROI_DIR, "labels")

# 归属规则：推荐 intersect（比center更鲁棒）
ASSIGN_RULE = "intersect"  # "intersect" or "center"

# 空标签处理：
# False：仍生成空txt（当负样本，通常OK）
# True ：没有线条就跳过写label（更严格，避免漏标污染）
SKIP_EMPTY_LABEL = False

# （可选）导出可训练YOLO数据集（images/labels + train/val 8:2）
EXPORT_DATASET = True
OUT_YOLO_ROOT = "/root/ultralytics-main/roi_line_yolo_dataset"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# 类别（线条检测）
CLASS_NAMES = ["line"]
# =======================================================


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def box_intersection_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return iw * ih

def belongs_to_roi(x1, y1, x2, y2, rx1, ry1, rx2, ry2):
    if ASSIGN_RULE == "center":
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)
    # intersect
    return box_intersection_area(x1, y1, x2, y2, rx1, ry1, rx2, ry2) > 0

def load_big_yolo_labels(txt_path):
    """读取大图线条YOLO标签 (cls cx cy w h), 归一化坐标"""
    items = []
    if not os.path.exists(txt_path):
        return items
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))  # 允许 0.0
            cx, cy, w, h = map(float, parts[1:])
            if any(math.isnan(v) or math.isinf(v) for v in [cx, cy, w, h]):
                continue
            items.append((cls, cx, cy, w, h))
    return items

def yolo_to_xyxy(cls, cx, cy, w, h, W, H):
    """YOLO归一化 -> 大图像素xyxy"""
    cx *= W; cy *= H; w *= W; h *= H
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return int(cls), x1, y1, x2, y2

def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    """ROI内像素xyxy -> ROI内YOLO归一化"""
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw /= w
    bh /= h
    return cx, cy, bw, bh

def write_data_yaml(out_root, class_names):
    yaml_path = os.path.join(out_root, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        for i, n in enumerate(class_names):
            f.write(f"  {i}: {n}\n")
    return yaml_path


# ------------------ Step 1: 大图分割裁剪 ------------------
def step1_crop_rois():
    ensure_dir(ROI_DIR)

    model = YOLO(WEIGHTS_ROI)

    csv_path = os.path.join(ROI_DIR, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_name", "roi_name", "roi_id", "x1", "y1", "x2", "y2", "conf"])

        imgs = sorted([p for p in os.listdir(IMG_DIR)
                       if p.lower().endswith((".png", ".jpg", ".jpeg"))])

        total_rois = 0
        for img_name in imgs:
            img_path = os.path.join(IMG_DIR, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print("[WARN] Skip unreadable:", img_path)
                continue

            H, W = img.shape[:2]
            r = model.predict(source=img, conf=CONF, iou=IOU, verbose=False)[0]

            if r.boxes is None or len(r.boxes) == 0:
                print("[WARN] No detections:", img_name)
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            # 按 x_center 排序（从左到右）
            idx = sorted(range(len(boxes)),
                         key=lambda i: (boxes[i][0] + boxes[i][2]) / 2.0)

            for roi_id, i in enumerate(idx, start=1):
                x1, y1, x2, y2 = boxes[i]
                c = float(confs[i])

                bw, bh = (x2 - x1), (y2 - y1)
                pad = PAD_RATIO * max(bw, bh)

                xx1 = int(clip(x1 - pad, 0, W - 1))
                yy1 = int(clip(y1 - pad, 0, H - 1))
                xx2 = int(clip(x2 + pad, 0, W - 1))
                yy2 = int(clip(y2 + pad, 0, H - 1))

                roi = img[yy1:yy2, xx1:xx2].copy()
                roi_name = f"{os.path.splitext(img_name)[0]}_roi{roi_id:03d}.png"
                roi_path = os.path.join(ROI_DIR, roi_name)
                cv2.imwrite(roi_path, roi)

                writer.writerow([img_name, roi_name, roi_id, xx1, yy1, xx2, yy2, f"{c:.6f}"])
                total_rois += 1

            print(f"{img_name}: saved {len(boxes)} rois")

    print("[Step1] Done. metadata.csv:", csv_path)
    return csv_path


# ------------------ Step 2: ROI标签生成 ------------------
def step2_generate_roi_labels(meta_csv_path):
    ensure_dir(ROI_LABEL_DIR)

    with open(meta_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # 按大图分组，减少重复读取
    by_img = {}
    for r in rows:
        by_img.setdefault(r["img_name"], []).append(r)

    total_roi = 0
    total_written = 0
    total_empty = 0
    missing_big_label = 0
    missing_big_img = 0

    for img_name, roi_rows in by_img.items():
        big_img_path = os.path.join(IMG_DIR, img_name)  # 这里用 IMG_DIR，确保一致
        big = cv2.imread(big_img_path)
        if big is None:
            print("[WARN] unreadable big image:", big_img_path)
            missing_big_img += len(roi_rows)
            continue
        H, W = big.shape[:2]

        big_label_path = os.path.join(BIG_LINE_LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")
        big_labels = load_big_yolo_labels(big_label_path)
        if not big_labels:
            print("[WARN] no big line labels:", big_label_path)
            missing_big_label += len(roi_rows)

        # 转像素框
        big_boxes = []
        for cls, cx, cy, bw, bh in big_labels:
            cls, x1, y1, x2, y2 = yolo_to_xyxy(cls, cx, cy, bw, bh, W, H)
            big_boxes.append((cls, x1, y1, x2, y2))

        for r in roi_rows:
            roi_name = r["roi_name"]
            rx1 = float(r["x1"]); ry1 = float(r["y1"])
            rx2 = float(r["x2"]); ry2 = float(r["y2"])
            roi_w = int(round(rx2 - rx1))
            roi_h = int(round(ry2 - ry1))

            out_txt = os.path.join(ROI_LABEL_DIR, os.path.splitext(roi_name)[0] + ".txt")

            roi_lines = []
            for cls, x1, y1, x2, y2 in big_boxes:
                if not belongs_to_roi(x1, y1, x2, y2, rx1, ry1, rx2, ry2):
                    continue

                # 裁剪到 ROI 边界（大图坐标系）
                ix1 = clip(x1, rx1, rx2)
                iy1 = clip(y1, ry1, ry2)
                ix2 = clip(x2, rx1, rx2)
                iy2 = clip(y2, ry1, ry2)

                # ROI内坐标
                lx1 = ix1 - rx1
                ly1 = iy1 - ry1
                lx2 = ix2 - rx1
                ly2 = iy2 - ry1

                yolo = xyxy_to_yolo(lx1, ly1, lx2, ly2, roi_w, roi_h)
                if yolo is None:
                    continue
                cx, cy, bw, bh = yolo
                if bw <= 0 or bh <= 0:
                    continue

                roi_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if len(roi_lines) == 0:
                total_empty += 1
                if SKIP_EMPTY_LABEL:
                    # 不写label（也可以写空文件，看你的训练策略）
                    continue

            # 写 label（可能为空）
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(roi_lines) + ("\n" if roi_lines else ""))

            total_roi += 1
            total_written += len(roi_lines)

    print("[Step2] Done. ROI label dir:", ROI_LABEL_DIR)
    print("[Step2] ROI files written:", total_roi, "total anns:", total_written)
    print("[Step2] Empty ROI labels:", total_empty)
    print("[Step2] Missing big images:", missing_big_img)
    print("[Step2] Missing big label files:", missing_big_label)


# ------------------ Step 3: 导出YOLO数据集(可选) ------------------
def step3_export_dataset():
    img_train_dir = os.path.join(OUT_YOLO_ROOT, "images", "train")
    img_val_dir   = os.path.join(OUT_YOLO_ROOT, "images", "val")
    lbl_train_dir = os.path.join(OUT_YOLO_ROOT, "labels", "train")
    lbl_val_dir   = os.path.join(OUT_YOLO_ROOT, "labels", "val")
    ensure_dir(img_train_dir); ensure_dir(img_val_dir)
    ensure_dir(lbl_train_dir); ensure_dir(lbl_val_dir)

    roi_imgs = sorted(glob.glob(os.path.join(ROI_DIR, "*.png")))
    if not roi_imgs:
        raise FileNotFoundError(f"No ROI png found in {ROI_DIR}")

    random.seed(RANDOM_SEED)
    random.shuffle(roi_imgs)
    n_train = int(len(roi_imgs) * TRAIN_RATIO)
    train_imgs = set(roi_imgs[:n_train])

    copied = 0
    missing_lbl = 0

    for img_path in roi_imgs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(ROI_LABEL_DIR, base + ".txt")

        subset = "train" if img_path in train_imgs else "val"
        if subset == "train":
            out_img_dir, out_lbl_dir = img_train_dir, lbl_train_dir
        else:
            out_img_dir, out_lbl_dir = img_val_dir, lbl_val_dir

        shutil.copy2(img_path, os.path.join(out_img_dir, os.path.basename(img_path)))

        # label：如果不存在，就写空文件保证对齐
        out_lbl = os.path.join(out_lbl_dir, base + ".txt")
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, out_lbl)
        else:
            missing_lbl += 1
            with open(out_lbl, "w", encoding="utf-8") as f:
                f.write("")

        copied += 1

    yaml_path = write_data_yaml(OUT_YOLO_ROOT, CLASS_NAMES)
    print("[Step3] Done. Exported:", copied, "Missing roi labels:", missing_lbl)
    print("[Step3] Dataset root:", OUT_YOLO_ROOT)
    print("[Step3] data.yaml:", yaml_path)


def main():
    ensure_dir(ROI_DIR)

    meta_csv = step1_crop_rois()                  # 1) 大图分割
    step2_generate_roi_labels(meta_csv)           # 2) ROI标签生成

    if EXPORT_DATASET:
        step3_export_dataset()                    # 3) 8:2导出YOLO数据集（可选）

if __name__ == "__main__":
    main()