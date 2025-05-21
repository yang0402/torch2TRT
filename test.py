import torch
import cv2
import numpy as np
import os
import torch_tensorrt  # 确保导入 torch_tensorrt

# 加载 TensorRT 模型
trt_path = "./output/real_esrgan.trt"
try:
    model_trt = torch.jit.load(trt_path).eval().cuda()
    print("TensorRT 模型加载成功！")
except Exception as e:
    print(f"加载 TensorRT 模型失败：{e}")
    exit(1)

def preprocess_image(img_path, side=256):
    """图像预处理：调整尺寸、归一化、pixel_unshuffle"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像：{img_path}")
    img = cv2.resize(img, (side, side))  # 调整大小为 side x side
    img = img[..., ::-1].transpose(2, 0, 1) / 255.0  # BGR -> RGB, HWC -> CHW
    x = torch.from_numpy(img).cuda().half().unsqueeze(0)  # FP16 + batch 维度

    # 手动实现 pixel_unshuffle (downscale_factor=2)
    b, c, h, w = x.shape
    h //= 2
    w //= 2
    x = x.view(b, c, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h, w)
    return x, img.shape[1], img.shape[2]  # 返回原始宽高

def postprocess_output(output, original_h, original_w):
    """后处理：将输出张量转回图像格式"""
    output = output.clamp_(0, 1).cpu().float().numpy()
    output = output[0].transpose(1, 2, 0)  # CHW -> HWC
    output = (output * 255.0).round().astype(np.uint8)  # 反归一化
    output = output[..., ::-1]  # RGB -> BGR
    return output

# 示例推理
if __name__ == "__main__":
    input_img_path = "./src/inputs/sr/long.jpg"
    output_img_path = "./result/output_trt.png"

    # Step 1: 图像预处理
    try:
        x, ori_h, ori_w = preprocess_image(input_img_path, side=256)
    except Exception as e:
        print(f"预处理失败：{e}")
        exit(1)

    # Step 2: 推理（TensorRT 模型）
    with torch.no_grad():
        try:
            y = model_trt(x)
        except Exception as e:
            print(f"推理失败：{e}")
            exit(1)

    # Step 3: 后处理并保存结果
    try:
        result = postprocess_output(y, ori_h * 2, ori_w * 2)
        cv2.imwrite(output_img_path, result)
        print(f"推理完成，结果已保存至：{os.path.abspath(output_img_path)}")
    except Exception as e:
        print(f"后处理失败：{e}")
        exit(1)