import os
import glob
import math

LABEL_ROOT = "/root/ultralytics-main/yolo_dataset/labels"
SUBDIRS = ["train", "val"]

def is_valid_num(x: float) -> bool:
    return (not math.isnan(x)) and (not math.isinf(x))

def fix_file(fp: str) -> tuple[int, int]:
    """return (kept_lines, dropped_lines)"""
    kept, dropped = 0, 0
    new_lines = []

    with open(fp, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 5:
            dropped += 1
            continue

        try:
            cls = int(float(parts[0]))  # 핵심：强制把 0.0 -> 0
            x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
        except Exception:
            dropped += 1
            continue

        if not all(is_valid_num(v) for v in [x, y, w, h]):
            dropped += 1
            continue

        # 可选：过滤异常归一化范围（允许略微越界，可按需收紧）
        if w <= 0 or h <= 0:
            dropped += 1
            continue

        new_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        kept += 1

    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

    return kept, dropped

def main():
    total_files = 0
    total_kept = 0
    total_dropped = 0

    for sd in SUBDIRS:
        pattern = os.path.join(LABEL_ROOT, sd, "*.txt")
        files = glob.glob(pattern)
        print(f"[{sd}] files: {len(files)}")
        for fp in files:
            total_files += 1
            kept, dropped = fix_file(fp)
            total_kept += kept
            total_dropped += dropped

    print("===================================")
    print(f"Processed label files: {total_files}")
    print(f"Kept lines: {total_kept}")
    print(f"Dropped bad lines: {total_dropped}")
    print("Done.")

if __name__ == "__main__":
    main()