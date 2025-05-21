import onnxruntime as ort
import numpy as np
import cv2
import os
import torch  # 用于 pixel_unshuffle

# 设置 ONNX 模型路径
onnx_path = "./output/real_esrgan.onnx"

# 确保模型文件存在
if not os.path.exists(onnx_path):
    raise FileNotFoundError(f"ONNX 模型文件不存在：{onnx_path}")

# 预处理函数：加载图像、调整大小、pixel_unshuffle
def preprocess_image(img_path, side=256):
    """
    预处理图像：调整大小、转换为 RGB、归一化、pixel_unshuffle。
    参数：
        img_path: 输入图像路径
        side: 调整后的图像尺寸（默认 256x256）
    返回：
        input_data: 预处理后的张量 (1, 12, side/2, side/2)
        ori_h, ori_w: 原始图像高度和宽度
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像：{img_path}")
    
    # 调整大小
    img = cv2.resize(img, (side, side))  # 调整为 side x side
    ori_h, ori_w = img.shape[:2]  # 保存原始尺寸
    
    # BGR 转 RGB，HWC 转 CHW，归一化
    img = img[..., ::-1].transpose(2, 0, 1) / 255.0  # (3, side, side)
    img = img.astype(np.float16)  # 转换为 float32
    
    # 转换为张量并添加 batch 维度
    x = torch.from_numpy(img).unsqueeze(0)  # (1, 3, side, side)
    
    # 手动实现 pixel_unshuffle (downscale_factor=2)
    b, c, h, w = x.shape
    h //= 2
    w //= 2
    x = x.view(b, c, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h, w)
    # 输出形状：(1, 12, side/2, side/2)
    
    return x.numpy(), ori_h, ori_w

# 后处理函数：将模型输出转换为图像
def postprocess_output(output, original_h, original_w, scale_factor=2):
    """
    后处理模型输出：反归一化、转换为 BGR、调整大小
    参数：
        output: 模型输出张量 (1, 3, height*scale_factor, width*scale_factor)
        original_h, original_w: 原始图像尺寸
        scale_factor: 超分辨率放大倍数（默认 2）
    返回：
        output_img: 处理后的图像 (BGR 格式)
    """
    # 确保输出在 [0, 1] 范围内
    output = np.clip(output, 0, 1)
    
    # 移除 batch 维度，CHW 转 HWC
    output = output[0].transpose(1, 2, 0)  # (height*scale_factor, width*scale_factor, 3)
    
    # 反归一化并转换为 uint8
    output = (output * 255.0).round().astype(np.uint8)
    
    # RGB 转 BGR（OpenCV 使用 BGR）
    output = output[..., ::-1]
    
    # 可选：调整到原始尺寸的放大版本
    output = cv2.resize(output, (original_w * scale_factor, original_h * scale_factor))
    
    return output

# 主程序
if __name__ == "__main__":
    # 输入图像路径和输出路径
    input_img_path = "./src/inputs/sr/long.jpg"  # 替换为你的输入图像
    output_img_path = "./result/output_onnx.png"
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    
    try:
        # 加载 ONNX 模型
        session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print("ONNX 模型加载成功！")
        
        # 预处理图像
        input_data, ori_h, ori_w = preprocess_image(input_img_path, side=256)
        print(f"输入形状: {input_data.shape}")
        
        # 执行推理
        outputs = session.run(None, {"input": input_data})
        output = outputs[0]  # 假设模型只有一个输出
        print(f"输出形状: {output.shape}")
        
        # 后处理输出
        result = postprocess_output(output, ori_h, ori_w, scale_factor=2)
        
        # 保存结果
        cv2.imwrite(output_img_path, result)
        print(f"推理完成，结果已保存至：{os.path.abspath(output_img_path)}")
        
    except Exception as e:
        print(f"推理失败：{e}")