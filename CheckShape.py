import torch

# 加载预训练权重
ckpt = torch.load("./src/pretrained_models/real_esrgan/RealESRGAN_x2plus.pth")

# 查看第一个卷积层的权重形状
print(ckpt['params_ema']['conv_first.weight'].shape)