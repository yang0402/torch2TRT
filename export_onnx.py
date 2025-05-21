import torch
# 导入 PyTorch 模块，用于模型操作和 ONNX 导出。

from test.common import prepare
# 从 test.common 模块导入 prepare 函数，用于准备模型和输入数据。

import os
# 导入 os 模块，用于文件和目录操作，例如创建目录。

# 设置输出路径
os.makedirs("output", exist_ok=True)
# 创建 output 目录，用于保存导出的 ONNX 模型，exist_ok=True 表示如果目录已存在不会报错。

# 准备模型和输入
model, x = prepare(side=256)
# 调用 prepare 函数，生成模型（可能是 RRDBNet）和输入张量 x。
# 参数 side=256 表示输入图像调整为 256x256。
# 返回值：model 是 PyTorch 模型，x 是输入张量，形状可能是 (1, 12, 256, 256)（假设 pixel_unshuffle 后通道数为 12）。

# ONNX 输出路径
onnx_path = "output/real_esrgan.onnx"
# 定义 ONNX 模型的保存路径为 output/real_esrgan.onnx。

# 导出为 ONNX 格式
torch.onnx.export(
    # 调用 torch.onnx.export 函数，将 PyTorch 模型导出为 ONNX 格式。
    
    model,
    # 第一个参数：要导出的 PyTorch 模型（例如 RRDBNet）。
    
    x,
    # 第二个参数：示例输入张量 x，用于追踪模型的计算图，形状如 (1, 12, 256, 256)。
    
    onnx_path,
    # 第三个参数：ONNX 模型的保存路径（output/real_esrgan.onnx）。
    
    export_params=True,  # 存储训练参数
    # 导出模型的权重和偏置到 ONNX 文件，生成可直接用于推理的模型。
    
    opset_version=13,    # ONNX 算子集版本
    # 使用 ONNX 算子集版本 13，确保兼容性并支持 PyTorch 的常见操作。
    
    do_constant_folding=True,  # 优化常量
    # 启用常量折叠，提前计算常量操作，优化模型大小和推理速度。
    
    input_names=['input'],     # 输入节点名称
    # 将模型的输入张量命名为 'input'，便于推理引擎识别。
    
    output_names=['output'],   # 输出节点名称
    # 将模型的输出张量命名为 'output'，便于推理引擎获取输出。
    
    dynamic_axes={
        # 定义动态维度，支持输入和输出的灵活形状。
        
        'input': {
            0: 'batch_size', 2: 'height', 3: 'width'  # 支持动态尺寸
            # 输入张量的第 0 维（批量大小）、第 2 维（高度）、第 3 维（宽度）是动态的。
            # 允许推理时使用不同批量大小和图像尺寸（例如 512x512 或 64x64）。
        },
        
        'output': {
            0: 'batch_size', 2: 'height', 3: 'width'
            # 输出张量的第 0 维（批量大小）、第 2 维（高度）、第 3 维（宽度）是动态的。
            # 输出尺寸通常与输入成比例（例如超分辨率模型放大 2 倍）。
        }
    }
)
#没有 dynamic_axes,ONNX 模型的输入和输出形状固定为导出时的示例输入形状,推理时必须使用完全相同的形状，否则报错。
#有 dynamic_axes,允许模型在推理时接受可变形状的输入，适合动态任务。
print(f"ONNX 模型已保存至：{onnx_path}")
# 打印成功信息，确认 ONNX 模型已保存到指定路径。