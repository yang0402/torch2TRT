import cv2       # OpenCV 库，用于图像读取与处理
import torch     # PyTorch 深度学习框架
import numpy as np  # 数值计算库，用于数组操作

def prepare(side=512):
    """加载并构建模型和输入张量"""
    
    # 获取一个超分辨率模型（如 Real-ESRGAN），并设置为评估模式（不启用 dropout/batchnorm 的训练行为）
    model = get_sr_model().eval()

    # 使用 OpenCV 读取图像，并调整大小到 side x side 像素
    img = cv2.resize(cv2.imread('./src/inputs/sr/long.jpg'), (side, side))
    
    # 将图像从 BGR 格式转换为 RGB 格式（OpenCV 默认是 BGR）
    # 并将维度从 HWC (Height, Width, Channels) 转换为 CHW (Channels, Height, Width)
    # 最后归一化到 [0, 1] 范围
    img = img[..., ::-1].transpose(2, 0, 1) / 255.0
    
    # 将 NumPy 数组转换为 PyTorch 张量
    # 并移动到 GPU 上，使用半精度浮点数（FP16），增加 batch 维度（unsqueeze）
    x = torch.from_numpy(img).cuda().half().unsqueeze(0)

    # 手动实现 pixel_unshuffle 操作（类似于降采样）
    # 输入形状为 [b, c, h, w]
    b, c, h, w = x.shape
    
    # 高度和宽度各缩小一半
    h //= 2
    w //= 2

    # 重新排列张量结构，模拟 pixel_unshuffle 效果：
    # 把每个 2x2 的像素块变成通道维度，从而通道数变为原来的 4 倍
    x = x.view(b, c, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h, w)

    # 返回模型和预处理后的输入张量
    return model, x


def get_sr_model():
    # 导入自定义的 RRDBNet 网络结构（Real-ESRGAN 使用的网络）
    from test.rrdb_net import RRDBNet
    
    # 创建 RRDBNet 模型实例
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)

    # 加载预训练模型权重（state_dict）文件
    load_net = torch.load('src/pretrained_models/real_esrgan/RealESRGAN_x2plus.pth')

    # 将加载的权重加载到模型中
    # 'params_ema' 是指数平均后的参数，通常在推理时使用
    model.load_state_dict(load_net['params_ema'], strict=True)

    # 设置为评估模式
    model.eval()
    
    # 移动模型到 GPU，并使用 FP16 半精度
    model = model.to('cuda').half()
    
    # 返回模型
    return model


def test(x, model, name):
    # 在无梯度计算环境下进行推理
    with torch.no_grad():
        y = model(x)
    
    # 打印输出张量的名称和形状
    print(name, y.shape)
    
    # 返回推理结果
    return y


def benchmark(model, x, warm_up=5, runs=50):
    # 在无梯度计算环境下进行性能测试
    with torch.no_grad():
        # 先运行几次以“热身”，防止首次运行影响性能统计
        for _ in range(warm_up):
            model(x)
        
        # 同步 GPU，确保所有操作完成
        torch.cuda.synchronize()

        # 存储每次推理时间
        timings = []
        
        # 进行多次推理测试
        for i in range(runs):
            s = time.time()      # 记录开始时间
            model(x)             # 推理
            torch.cuda.synchronize()  # 等待 GPU 完成当前任务
            timings.append(time.time() - s)  # 记录耗时

            # 每运行10次打印一次中间结果
            if i % 10 == 0:
                print(f"Iteration {i+1}/{runs}, time: {np.mean(timings[-10:]) * 1000:.2f} ms")
        
        # 打印平均推理时间（单位：毫秒）
        print(f"Avg time: {np.mean(timings) * 1000:.2f} ms")