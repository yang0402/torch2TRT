# 导入必要的库
import torch
import torch_tensorrt  # Torch-TensorRT 接口，用于将 PyTorch 模型编译为 TensorRT 模型

# 从测试工具模块导入 prepare 函数，用于准备模型和输入数据
from test.common import prepare

# 使用 prepare 函数加载一个预定义的模型 model 和示例输入 x（即固定图片进行预处理后的张量）
# 参数 side=256 表示输入图像尺寸为 256x256,会把固定图片处理成256x256。
model, x = prepare(side=256)

# 定义编译设置，用于配置 TensorRT 编译参数
compile_settings = {
    "inputs": [
        torch_tensorrt.Input(
            min_shape=[1, 12, 64, 64],   # 最小输入形状
            opt_shape=[1, 12, 128, 128],  # 最优输入形状（TensorRT 会对此进行优化）
            max_shape=[1, 12, 256, 256],  # 最大输入形状
            dtype=torch.half              # 输入精度为 FP16
        )
    ],
    "enabled_precisions": {torch.half},  # 启用 FP16 精度加速推理
    "truncate_long_and_double": True     # 将 long/double 类型转换为 int/float
}

# 在无梯度计算模式下跟踪模型
with torch.no_grad():
    # 使用 torch.jit.trace 对模型进行跟踪，生成 TorchScript 模型
    #torch.jit.trace(model, x) 是通过传入一个示例输入 x，"录制"模型的一次前向传播过程，
    #从而生成一个脱离 Python 的 TorchScript 静态图模型（.pt），便于后续部署和优化。
    #即.pt 包含完整模型结构和参数，可以在 C++ 上运行。
    #.pth只有参数而无结构。
    traced_model = torch.jit.trace(model, x)

# 使用 Torch-TensorRT 编译器将 TorchScript 模型编译为 TensorRT 模型
print("开始编译 TensorRT 模型...")
#** 是一个解包操作符（unpacking operator），它用于将字典（dictionary）中的键值对作为关键字参数传递给函数
model_trt = torch_tensorrt.compile(traced_model, **compile_settings)
print("TensorRT 模型编译完成")

# 定义保存路径
trt_path = "output/real_esrgan.trt"

# 将编译后的 TensorRT 模型保存到指定路径
torch.jit.save(model_trt, trt_path)
print(f"TensorRT 模型已保存至：{trt_path}")

# 测试模型推理结果
y = model(x)               # 原始模型推理结果
y_trt = model_trt(x)       # TensorRT 模型推理结果

# 计算两个输出之间的最大绝对误差，并打印出来
print("最大误差:", torch.max(torch.abs(y - y_trt)).item())