1. pixel_unshuffle 是一种图像预处理操作，它将空间维度（H, W）上的像素“压缩”到通道维度上。即：空间尺寸缩小为原来的一半，通道数变为原来的 4 倍。
2. 可以输入min_shape到max_shape之间任意大小的分辨率,分辨率符合 pixel_unshuffle 的要求（如能被 2 整除）。如果图片分辨率不符合要求则需要进行预处理。即模型是否能处理高清图像（如 512x512、2K），完全取决于你在导出 TensorRT 模型时设置的 min_shape, opt_shape, max_shape。与你训练或测试时使用的图像大小无关。
3. 超分几倍是由模型结构和模型参数决定的。本项目是2倍,导出来的TRT最高支持256*256，因此当您输入2k的时候，需要resized到max_shape以下，最后输出resize*2的分辨率图片。
4. 建议使用UV来配置环境，UV会自动处理依赖关系。但是当您下载某个依赖时，他会下载最新版同时会改变其他依赖版本以此来符合您下载的那个依赖。建议使用uv pip install torch-tensorrt --no-upgrade在不改变当前环境的情况下下载您所需要的torch-tensorrt或者其他依赖。PIP并不会自动处理依赖关系，只会默认下载最新版。
5. torch-tensorrt的GITHUB地址https://github.com/pytorch/TensorRT
6. 项目参考地址https://blog.csdn.net/qq_29598161/article/details/121765839
7. python export_tensorrt.py 导出TRT模型
8. python test.py 测试导出的TRT模型