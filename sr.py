import os
import cv2
import torch

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# --------------------------------------------------------
# ★★ 手动修改下面三个路径即可 ★★
# --------------------------------------------------------
INPUT_IMAGE = "/root/Real-ESRGAN/Resistance193_roi003.png"         # 输入低分辨率图像
OUTPUT_IMAGE = "/root/Real-ESRGAN/output/resistance_SRx4.png"      # 输出高分辨率图像
MODEL_PATH = "/root/Real-ESRGAN/weights/RealESRGAN_x4plus.pth"     # 预训练模型路径
UPSCALE = 4                                                         # 放大倍数
# --------------------------------------------------------


def super_resolve(input_path, output_path, model_path, scale=4, outscale=4):
    # 1) 检查输入图像是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入图像不存在: {input_path}")

    # 2) 检查模型权重是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重不存在: {model_path}")

    # 3) 创建输出目录
    out_dir = os.path.dirname(output_path)
    if out_dir and (not os.path.exists(out_dir)):
        os.makedirs(out_dir, exist_ok=True)

    # 4) 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # 5) 构建 RRDBNet 模型（与 RealESRGAN_x4plus 配置一致）
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale
    )

    # 6) 构建 RealESRGANer 封装器
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,          # 不切 tile（小图/显存够时可以为 0）
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()   # GPU 上用半精度节省显存
    )

    # 7) 读取图像（BGR）
    print(f"[INFO] 正在读取图像: {input_path}")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"OpenCV 无法读取图像: {input_path}")

    # 8) 超分推理
    print("[INFO] 正在执行超分辨率推理...")
    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        print("[ERROR] 推理过程中出现显存/尺寸问题，可以尝试减小 tile 大小。")
        raise e

    # 9) 保存结果（BGR）
    cv2.imwrite(output_path, output)
    print(f"[SUCCESS] 已保存超分图像 → {output_path}")


if __name__ == "__main__":
    super_resolve(INPUT_IMAGE, OUTPUT_IMAGE, MODEL_PATH, scale=UPSCALE, outscale=UPSCALE)