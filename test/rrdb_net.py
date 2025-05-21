import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.archs.arch_util import default_init_weights, make_layer
# 导入必要的 PyTorch 模块和工具：
# - torch: PyTorch 主模块。
# - nn: PyTorch 的神经网络模块，简化为 nn。
# - F: PyTorch 的功能模块，用于激活函数、插值等操作。
# - default_init_weights, make_layer: 从 basicsr 库导入的工具函数，用于权重初始化和层堆叠。

class ResidualDenseBlock(nn.Module):
    # 定义 ResidualDenseBlock 类，继承自 nn.Module，是一个残差密集块（RDB），用于特征提取。
    def __init__(self, num_feat=64, num_grow_ch=32):
        # 初始化函数，设置 ResidualDenseBlock 的参数。
        # num_feat: 输入和输出的特征通道数，默认为 64。
        # num_grow_ch: 每层卷积的增长通道数，默认为 32。
        super(ResidualDenseBlock, self).__init__()
        # 调用父类 nn.Module 的初始化方法。

        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        # 定义第一个 2D 卷积层，输入通道数为 num_feat，输出通道数为 num_grow_ch，卷积核大小 3x3，步幅 1，填充 1。
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        # 定义第二个卷积层，输入通道数为 num_feat + num_grow_ch（拼接后的通道数），输出通道数为 num_grow_ch。
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        # 定义第三个卷积层，输入通道数为 num_feat + 2 * num_grow_ch，输出通道数为 num_grow_ch。
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        # 定义第四个卷积层，输入通道数为 num_feat + 3 * num_grow_ch，输出通道数为 num_grow_ch。
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        # 定义第五个卷积层，输入通道数为 num_feat + 4 * num_grow_ch，输出通道数为 num_feat，恢复到原始特征通道数。

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 定义 LeakyReLU 激活函数，负斜率为 0.2，inplace=True 表示原地操作以节省内存。
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        # 使用 default_init_weights 函数对所有卷积层的权重进行初始化，缩放因子为 0.1。

    def forward(self, x):
        # 前向传播函数，定义数据通过 ResidualDenseBlock 的处理流程。
        # x: 输入张量，形状为 (batch_size, num_feat, height, width)。
        x1 = self.lrelu(self.conv1(x))
        # 第一个卷积层处理输入 x，经过 LeakyReLU 激活，得到 x1，形状为 (batch_size, num_grow_ch, height, width)。
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        # 将输入 x 和 x1 在通道维度（dim=1）拼接，输入到第二个卷积层，经过 LeakyReLU 激活，得到 x2。
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # 拼接 x, x1, x2，输入到第三个卷积层，经过 LeakyReLU 激活，得到 x3。
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # 拼接 x, x1, x2, x3，输入到第四个卷积层，经过 LeakyReLU 激活，得到 x4。
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # 拼接 x, x1, x2, x3, x4，输入到第五个卷积层，得到 x5，形状恢复为 (batch_size, num_feat, height, width)。
        return x5 * 0.2 + x
        # 将 x5 乘以 0.2（残差缩放因子）并与输入 x 相加，构成残差连接，返回最终输出。

class RRDB(nn.Module):
    # 定义 RRDB（Residual in Residual Dense Block）类，继承自 nn.Module，包含三个 ResidualDenseBlock。
    def __init__(self, num_feat, num_grow_ch=32):
        # 初始化函数，设置 RRDB 的参数。
        # num_feat: 输入和输出的特征通道数。
        # num_grow_ch: 每个 RDB 的增长通道数，默认为 32。
        super(RRDB, self).__init__()
        # 调用父类 nn.Module 的初始化方法。

        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        # 定义第一个 ResidualDenseBlock。
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        # 定义第二个 ResidualDenseBlock。
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
        # 定义第三个 ResidualDenseBlock。

    def forward(self, x):
        # 前向传播函数，定义数据通过 RRDB 的处理流程。
        # x: 输入张量，形状为 (batch_size, num_feat, height, width)。
        out = self.rdb1(x)
        # 输入 x 通过第一个 RDB，得到中间特征 out。
        out = self.rdb2(out)
        # 中间特征 out 通过第二个 RDB，更新 out。
        out = self.rdb3(out)
        # 中间特征 out 通过第三个 RDB，更新 out。
        return out * 0.2 + x
        # 将最终输出 out 乘以 0.2（残差缩放因子）并与输入 x 相加，构成残差连接，返回最终输出。

class RRDBNet(nn.Module):
    # 定义 RRDBNet 类，继承自 nn.Module，是一个基于 RRDB 的完整网络，用于超分辨率等任务。
    def __init__(self, num_in_ch, num_out_ch, num_feat=64, num_block=23, num_grow_ch=32):
        # 初始化函数，设置 RRDBNet 的参数。
        # num_in_ch: 输入图像的通道数（通常为 3，RGB 图像）。
        # num_out_ch: 输出图像的通道数（通常为 3，RGB 图像）。
        # num_feat: 中间特征的通道数，默认为 64。
        # num_block: RRDB 模块的数量，默认为 23。
        # num_grow_ch: 每个 RDB 的增长通道数，默认为 32。
        super(RRDBNet, self).__init__()
        # 调用父类 nn.Module 的初始化方法。

        num_in_ch = num_in_ch * 4  # 因为输入已经是 pixel_unshuffle 后的结果，所以通道数 ×4
        # 输入通道数乘以 4，因为输入图像经过 pixel_unshuffle 操作，通道数增加 4 倍（用于超分辨率）。
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # 定义第一个卷积层，将 pixel_unshuffle 后的输入（num_in_ch * 4 通道）转换为 num_feat 通道，卷积核 3x3，步幅 1，填充 1。
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        # 使用 make_layer 函数创建 num_block 个 RRDB 模块堆叠，组成网络的主体部分。
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # 定义主体部分的最后一个卷积层，保持通道数为 num_feat，卷积核 3x3，步幅 1，填充 1。

        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # 定义第一个上采样卷积层，保持通道数为 num_feat，卷积核 3x3，步幅 1，填充 1。
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # 定义第二个上采样卷积层，保持通道数为 num_feat，卷积核 3x3，步幅 1，填充 1。
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # 定义高分辨率卷积层，保持通道数为 num_feat，卷积核 3x3，步幅 1，填充 1。
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        # 定义最后一个卷积层，将特征通道数 num_feat 转换为输出通道数 num_out_ch，卷积核 3x3，步幅 1，填充 1。

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 定义 LeakyReLU 激活函数，负斜率为 0.2，inplace=True 表示原地操作以节省内存。

    def forward(self, x):
        # 前向传播函数，定义数据通过 RRDBNet 的处理流程。
        # x: 输入张量，形状为 (batch_size, num_in_ch * 4, height/2, width/2)，假设输入已进行 pixel_unshuffle。
        feat = self.conv_first(x)
        # 输入 x 通过第一个卷积层，得到初始特征 feat，形状为 (batch_size, num_feat, height/2, width/2)。
        body_feat = self.conv_body(self.body(feat))
        # 初始特征 feat 通过 RRDB 主体部分（多个 RRDB 模块），再经过 conv_body 卷积层，得到 body_feat。
        feat = feat + body_feat
        # 将 body_feat 与初始特征 feat 相加，构成残差连接，更新 feat。

        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        # 对 feat 进行 2 倍上采样（最近邻插值），通过 conv_up1 卷积层和 LeakyReLU 激活，得到上采样特征。
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        # 再次进行 2 倍上采样，通过 conv_up2 卷积层和 LeakyReLU 激活，进一步放大特征。
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        # 上采样特征通过 conv_hr 卷积层和 LeakyReLU 激活，再通过 conv_last 卷积层，得到最终输出 out。
        # out 的形状为 (batch_size, num_out_ch, height*2, width*2)，即输出分辨率放大 4 倍。
        return out
        # 返回最终输出张量。