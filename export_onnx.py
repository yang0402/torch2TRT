import torch
from test.common import prepare
import os

# 设置输出路径
os.makedirs("output", exist_ok=True)

# 准备模型和输入
model, x = prepare(side=256)

# ONNX 输出路径
onnx_path = "output/real_esrgan.onnx"

# 导出为 ONNX 格式
torch.onnx.export(
    model,
    x,
    onnx_path,
    export_params=True,  # 存储训练参数
    opset_version=13,    # ONNX 算子集版本
    do_constant_folding=True,  # 优化常量
    input_names=['input'],     # 输入节点名称
    output_names=['output'],   # 输出节点名称
    dynamic_axes={
        'input': {
            0: 'batch_size', 2: 'height', 3: 'width'  # 支持动态尺寸
        },
        'output': {
            0: 'batch_size', 2: 'height', 3: 'width'
        }
    }
)

print(f"ONNX 模型已保存至：{onnx_path}")