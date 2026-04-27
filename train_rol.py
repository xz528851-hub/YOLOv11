from ultralytics import YOLO
import os

# 你的ROI线条数据集yaml（你导出的 data.yaml）
DATA_YAML = "/root/ultralytics-main/ultralytics/cfg/datasets/line_coco.yaml"

# 预训练权重：小目标推荐从 s 起步（n太轻容易欠拟合）
MODEL_WEIGHTS = "yolo11m.pt"   # 可换 yolo11m.pt 追求更强精度

PROJECT = "/root/ultralytics-main/runs_rol"

def main():
    model = YOLO(MODEL_WEIGHTS)

    results = model.train(
        data=DATA_YAML,

        # 对细线条目标，建议 1024 或 1280
        imgsz=1280,

        epochs=250,
        batch=8,           # 显存够可改16；OOM就改4
        workers=4,

        optimizer="AdamW",
        lr0=0.003,
        lrf=0.01,
        weight_decay=0.01,
        warmup_epochs=5.0,

        # 建议保持长宽比，减少拉伸导致线条变形
        rect=True,

        # mosaic 前期可以用一点，后期关闭更利于精定位
        mosaic=0.5,
        close_mosaic=20,
        mixup=0.0,

        # 工业图像增强要克制
        hsv_h=0.01,
        hsv_s=0.30,
        hsv_v=0.30,
        degrees=0.0,
        translate=0.03,
        scale=0.20,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,

        # 训练细节
        patience=50,
        pretrained=True,
        amp=True,
        cache="ram",       # 内存不够改 False
        plots=True,

        # 输出
        project=PROJECT,
        name="yolo11_line_roi",
        exist_ok=True
    )

    save_dir = os.path.join(PROJECT, "yolo11_line_roi")
    print("Training finished.")
    print("Best:", os.path.join(save_dir, "weights", "best.pt"))
    print("Last:", os.path.join(save_dir, "weights", "last.pt"))

if __name__ == "__main__":
    main()