# test_cbam_simple.py
# 简单的CBAM模块测试，不使用pytest

import torch
import torch.nn as nn
import sys
import os

# 添加路径，确保能导入ultralytics模块
sys.path.append('D:/deep_learning/ultralytics-8.3.163')


class CBAM(nn.Module):
    """CBAM注意力模块"""

    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 通道注意力
        out = x * self.channel_attention(x)
        # 空间注意力
        out = out * self.spatial_attention(out)
        return out


class ChannelAttention(nn.Module):
    """CBAM通道注意力模块"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """CBAM空间注意力模块"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


def test_cbam_module():
    """测试CBAM模块的基本功能"""
    print("=" * 50)
    print("测试CBAM注意力模块")
    print("=" * 50)

    # 测试1：创建模块
    print("\n1. 创建CBAM模块...")
    cbam = CBAM(channels=64)
    print(f"   模块创建成功: {cbam}")

    # 测试2：前向传播
    print("\n2. 测试前向传播...")
    x = torch.randn(2, 64, 32, 32)  # [batch, channels, height, width]
    print(f"   输入形状: {x.shape}")

    output = cbam(x)
    print(f"   输出形状: {output.shape}")

    # 测试3：验证形状一致性
    if x.shape == output.shape:
        print(f"   ✓ 输入输出形状一致")
    else:
        print(f"   ✗ 输入输出形状不一致!")
        return False

    # 测试4：验证参数数量
    print("\n3. 验证参数数量...")
    total_params = sum(p.numel() for p in cbam.parameters())
    print(f"   总参数数量: {total_params:,}")

    # 测试5：梯度测试
    print("\n4. 测试梯度回传...")
    loss = output.mean()
    loss.backward()

    # 检查是否有梯度
    has_gradients = False
    for name, param in cbam.named_parameters():
        if param.grad is not None:
            has_gradients = True
            print(f"   ✓ 参数 '{name}' 有梯度")
            break

    if has_gradients:
        print("   ✓ 梯度回传正常")
    else:
        print("   ✗ 没有检测到梯度!")
        return False

    # 测试6：不同输入的兼容性
    print("\n5. 测试不同输入形状...")
    test_shapes = [
        (1, 64, 16, 16),  # 小图
        (4, 64, 32, 32),  # 批量大小变化
        (2, 64, 64, 64),  # 大图
        (1, 128, 32, 32),  # 通道数变化
    ]

    for i, shape in enumerate(test_shapes, 1):
        try:
            x_test = torch.randn(*shape)
            out_test = cbam(x_test)
            if x_test.shape == out_test.shape:
                print(f"   ✓ 测试 {i}: 形状 {shape} 通过")
            else:
                print(f"   ✗ 测试 {i}: 形状 {shape} 失败")
                return False
        except Exception as e:
            print(f"   ✗ 测试 {i}: 形状 {shape} 出错 - {e}")
            return False

    print("\n" + "=" * 50)
    print("所有测试通过！CBAM模块工作正常")
    print("=" * 50)
    return True


def test_cbam_integration():
    """测试CBAM在YOLO中的集成"""
    print("\n\n" + "=" * 50)
    print("测试CBAM在模型中的集成")
    print("=" * 50)

    try:
        # 尝试导入YOLO
        from ultralytics import YOLO
        print("✓ 成功导入ultralytics")

        # 测试YAML配置文件
        yaml_path = "yolov8n-cbam.yaml"
        if os.path.exists(yaml_path):
            print(f"✓ 找到YAML配置文件: {yaml_path}")

            # 尝试解析配置文件
            import yaml
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if 'backbone' in config and 'head' in config:
                print("✓ YAML配置文件格式正确")

                # 检查是否包含CBAM模块
                has_cbam = False
                for layer in config['backbone']:
                    if isinstance(layer, list) and len(layer) > 2:
                        if 'C2f_CBAM' in str(layer[2]):
                            has_cbam = True
                            break

                if has_cbam:
                    print("✓ 配置文件中包含C2f_CBAM模块")
                else:
                    print("⚠ 配置文件中未找到C2f_CBAM模块")
            else:
                print("✗ YAML配置文件格式不正确")

        else:
            print(f"✗ 未找到YAML配置文件: {yaml_path}")

    except ImportError as e:
        print(f"✗ 导入错误: {e}")
    except Exception as e:
        print(f"✗ 其他错误: {e}")


def main():
    """主函数"""
    print("CBAM注意力模块测试")
    print("当前工作目录:", os.getcwd())
    print("Python版本:", sys.version)
    print("PyTorch版本:", torch.__version__)
    print("CUDA可用:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA版本:", torch.version.cuda)
        print("GPU设备:", torch.cuda.get_device_name(0))

    # 运行测试
    test_result = test_cbam_module()

    if test_result:
        test_cbam_integration()

    return test_result


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)