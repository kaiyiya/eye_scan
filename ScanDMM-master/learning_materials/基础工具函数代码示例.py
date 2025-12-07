"""
基础工具函数代码示例
====================

这个文件包含了suppor_lib.py中所有基础工具函数的详细示例和可视化。
帮助你理解360°图像处理中的坐标系统转换。

运行方式：
1. 在Jupyter Notebook中运行：exec(open('基础工具函数代码示例.py', encoding='utf-8').read())
2. 或者在Python中直接运行：python 基础工具函数代码示例.py
"""

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import platform

# 设置matplotlib中文字体，解决中文显示问题
def setup_chinese_font():
    """设置matplotlib中文字体"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统常用中文字体（按优先级排序）
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'SimSun']
    elif system == 'Darwin':  # macOS
        chinese_fonts = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Heiti SC']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
    
    # 设置字体列表，matplotlib会自动选择第一个可用的字体
    plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    print(f"已设置中文字体列表: {chinese_fonts[:3]}... (系统: {system})")

# 立即设置字体
setup_chinese_font()

# 导入项目中的实际函数
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suppor_lib import sphere2xyz, xyz2sphere, sphere2plane, plane2sphere, xyz2plane

pi = math.pi

print("=" * 80)
print("基础工具函数代码示例")
print("=" * 80)
print("\n本示例将帮助你理解360°图像处理中的坐标系统转换\n")

# ============================================================================
# 第一部分：理解坐标系统
# ============================================================================

print("\n" + "=" * 80)
print("第一部分：理解三种坐标系统")
print("=" * 80)

print("""
360°图像处理中涉及三种坐标系统：

1. 球面坐标 (Sphere Coordinates)
   - 格式: (lat, lon) 或 (纬度, 经度)
   - 范围: lat ∈ [-90°, 90°], lon ∈ [-180°, 180°]
   - 含义: 地球表面的经纬度坐标
   - 示例: (0°, 0°) 表示赤道和本初子午线的交点

2. 3D单位球坐标 (3D Unit Sphere Coordinates)
   - 格式: (x, y, z)
   - 范围: x² + y² + z² = 1 (单位球)
   - 含义: 3D空间中的点，位于单位球面上
   - 用途: 在3D空间中表示注视方向

3. 平面坐标 (Plane Coordinates)
   - 格式: (x, y) 归一化到 [0, 1] 或像素坐标
   - 范围: x ∈ [0, 1], y ∈ [0, 1] (归一化)
   - 含义: 等距柱状投影图像中的像素位置
   - 用途: 在2D图像上表示位置
""")

# ============================================================================
# 第二部分：sphere2xyz - 球面坐标转3D坐标
# ============================================================================

print("\n" + "=" * 80)
print("第二部分：sphere2xyz() - 球面坐标转3D坐标")
print("=" * 80)

print("\n函数签名: sphere2xyz(sphere_cord)")
print("输入: (lat, lon) 形状 = (n, 2)")
print("输出: (x, y, z) 形状 = (n, 3)")
print("\n数学公式:")
print("  lat_rad = lat * π / 180")
print("  lon_rad = lon * π / 180")
print("  x = cos(lat_rad) * cos(lon_rad)")
print("  y = cos(lat_rad) * sin(lon_rad)")
print("  z = sin(lat_rad)")

# 测试关键点
test_points = torch.tensor([
    [0.0, 0.0],  # 赤道，0度经度 (应该对应 (1, 0, 0))
    [90.0, 0.0],  # 北极 (应该对应 (0, 0, 1))
    [-90.0, 0.0],  # 南极 (应该对应 (0, 0, -1))
    [0.0, 90.0],  # 赤道，东经90度 (应该对应 (0, 1, 0))
    [0.0, -90.0],  # 赤道，西经90度 (应该对应 (0, -1, 0))
    [45.0, 45.0],  # 东北方向
])

print("\n测试关键点:")
print("-" * 80)
xyz_points = sphere2xyz(test_points)
for i, (sp, xyz) in enumerate(zip(test_points, xyz_points)):
    norm = torch.norm(xyz).item()
    print(
        f"点{i + 1}: 球面({sp[0]:6.1f}°, {sp[1]:6.1f}°) -> 3D({xyz[0]:7.4f}, {xyz[1]:7.4f}, {xyz[2]:7.4f}) | 模长: {norm:.6f}")

# 验证单位球
print("\n验证单位球性质:")
print("-" * 80)
norms = torch.norm(xyz_points, dim=1)
print(f"所有点的模长: {norms.tolist()}")
print(f"是否都在单位球上: {torch.allclose(norms, torch.ones_like(norms))}")

# ============================================================================
# 第三部分：xyz2sphere - 3D坐标转球面坐标
# ============================================================================

print("\n" + "=" * 80)
print("第三部分：xyz2sphere() - 3D坐标转球面坐标")
print("=" * 80)

print("\n函数签名: xyz2sphere(threeD_cord)")
print("输入: (x, y, z) 形状 = (n, 3)")
print("输出: (lat, lon) 形状 = (n, 2)")
print("\n数学公式:")
print("  lon = atan2(y, x) * 180 / π")
print("  lat = atan2(z, sqrt(x² + y²)) * 180 / π")

# 测试：验证往返转换
print("\n测试往返转换 (sphere -> xyz -> sphere):")
print("-" * 80)
original_sphere = torch.tensor([[30.0, 45.0], [-45.0, 120.0], [60.0, -90.0]])
xyz_converted = sphere2xyz(original_sphere)
sphere_recovered = xyz2sphere(xyz_converted)

for i, (orig, recovered) in enumerate(zip(original_sphere, sphere_recovered)):
    error_lat = abs(orig[0] - recovered[0]).item()
    error_lon = abs(orig[1] - recovered[1]).item()
    # 处理经度的周期性（-180和180是同一个点）
    if error_lon > 180:
        error_lon = 360 - error_lon
    print(
        f"点{i + 1}: 原始({orig[0]:6.1f}°, {orig[1]:6.1f}°) -> 恢复({recovered[0]:6.1f}°, {recovered[1]:6.1f}°) | 误差: ({error_lat:.4f}°, {error_lon:.4f}°)")

# ============================================================================
# 第四部分：sphere2plane - 球面坐标转平面坐标
# ============================================================================

print("\n" + "=" * 80)
print("第四部分：sphere2plane() - 球面坐标转平面坐标")
print("=" * 80)

print("\n函数签名: sphere2plane(sphere_cord, height_width=None)")
print("输入: (lat, lon) 形状 = (n, 2)")
print("输出: (y, x) 形状 = (n, 2)  # 注意：返回的是(y, x)不是(x, y)")
print("\n数学公式 (归一化到[0,1]):")
print("  y = (lat + 90) / 180")
print("  x = (lon + 180) / 360")
print("\n数学公式 (像素坐标):")
print("  y = (lat + 90) / 180 * height")
print("  x = (lon + 180) / 360 * width")

# 测试归一化坐标
print("\n测试归一化坐标 [0, 1]:")
print("-" * 80)
test_sphere = torch.tensor([
    [0.0, 0.0],  # 赤道中心 -> (0.5, 0.5)
    [90.0, 0.0],  # 北极 -> (1.0, 0.5)
    [-90.0, 0.0],  # 南极 -> (0.0, 0.5)
    [0.0, 180.0],  # 赤道，东经180度 -> (0.5, 1.0)
    [0.0, -180.0],  # 赤道，西经180度 -> (0.5, 0.0)
])
plane_normalized = sphere2plane(test_sphere)
for i, (sp, pl) in enumerate(zip(test_sphere, plane_normalized)):
    print(f"点{i + 1}: 球面({sp[0]:6.1f}°, {sp[1]:6.1f}°) -> 平面(y={pl[0]:.3f}, x={pl[1]:.3f})")

# 测试像素坐标
print("\n测试像素坐标 (假设图像尺寸 128x256):")
print("-" * 80)
height, width = 128, 256
plane_pixel = sphere2plane(test_sphere, (height, width))
for i, (sp, pl) in enumerate(zip(test_sphere, plane_pixel)):
    print(f"点{i + 1}: 球面({sp[0]:6.1f}°, {sp[1]:6.1f}°) -> 像素(y={pl[0]:.1f}, x={pl[1]:.1f})")

# ============================================================================
# 第五部分：plane2sphere - 平面坐标转球面坐标
# ============================================================================

print("\n" + "=" * 80)
print("第五部分：plane2sphere() - 平面坐标转球面坐标")
print("=" * 80)

print("\n函数签名: plane2sphere(plane_cord, height_width=None)")
print("输入: (y, x) 形状 = (n, 2)  # 注意：输入是(y, x)")
print("输出: (lat, lon) 形状 = (n, 2)")
print("\n数学公式 (从归一化坐标):")
print("  lat = (y - 0.5) * 180")
print("  lon = (x - 0.5) * 360")
print("\n数学公式 (从像素坐标):")
print("  lat = (y / height - 0.5) * 180")
print("  lon = (x / width - 0.5) * 360")

# 测试往返转换
print("\n测试往返转换 (sphere -> plane -> sphere):")
print("-" * 80)
original_sphere = torch.tensor([[30.0, 45.0], [-45.0, 120.0]])
plane_converted = sphere2plane(original_sphere)
sphere_recovered = plane2sphere(plane_converted)

for i, (orig, recovered) in enumerate(zip(original_sphere, sphere_recovered)):
    error_lat = abs(orig[0] - recovered[0]).item()
    error_lon = abs(orig[1] - recovered[1]).item()
    if error_lon > 180:
        error_lon = 360 - error_lon
    print(
        f"点{i + 1}: 原始({orig[0]:6.1f}°, {orig[1]:6.1f}°) -> 恢复({recovered[0]:6.1f}°, {recovered[1]:6.1f}°) | 误差: ({error_lat:.4f}°, {error_lon:.4f}°)")

# ============================================================================
# 第六部分：xyz2plane - 3D坐标转平面坐标
# ============================================================================

print("\n" + "=" * 80)
print("第六部分：xyz2plane() - 3D坐标转平面坐标")
print("=" * 80)

print("\n函数签名: xyz2plane(threeD_cord, height_width=None)")
print("输入: (x, y, z) 形状 = (n, 3)")
print("输出: (y, x) 形状 = (n, 2)")
print("\n实现: 先转球面坐标，再转平面坐标")
print("  xyz -> sphere (xyz2sphere) -> plane (sphere2plane)")

# 测试完整转换链
print("\n测试完整转换链 (sphere -> xyz -> plane):")
print("-" * 80)
test_sphere = torch.tensor([[0.0, 0.0], [45.0, 90.0], [-30.0, -45.0]])
xyz_intermediate = sphere2xyz(test_sphere)
plane_final = xyz2plane(xyz_intermediate)

for i, (sp, xyz, pl) in enumerate(zip(test_sphere, xyz_intermediate, plane_final)):
    print(
        f"点{i + 1}: 球面({sp[0]:6.1f}°, {sp[1]:6.1f}°) -> 3D({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) -> 平面(y={pl[0]:.3f}, x={pl[1]:.3f})")

# ============================================================================
# 第七部分：可视化示例
# ============================================================================

print("\n" + "=" * 80)
print("第七部分：可视化示例")
print("=" * 80)

print("\n生成可视化图像...")

# 创建图形
fig = plt.figure(figsize=(16, 10))

# 子图1: 3D单位球上的点
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
sample_sphere = torch.tensor([
    [0, 0], [30, 0], [60, 0], [90, 0],
    [0, 45], [0, 90], [0, 135], [0, 180],
    [45, 45], [-45, -45]
])
sample_xyz = sphere2xyz(sample_sphere)
ax1.scatter(sample_xyz[:, 0], sample_xyz[:, 1], sample_xyz[:, 2], c='red', s=50)
# 绘制单位球
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='blue')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D单位球上的点')

# 子图2: 等距柱状投影网格
ax2 = fig.add_subplot(2, 3, 2)
lat_grid = np.linspace(-90, 90, 19)
lon_grid = np.linspace(-180, 180, 37)
for lat in lat_grid:
    sphere_line = torch.tensor([[lat, lon] for lon in lon_grid])
    plane_line = sphere2plane(sphere_line)
    ax2.plot(plane_line[:, 1].numpy(), plane_line[:, 0].numpy(), 'b-', alpha=0.3, linewidth=0.5)
for lon in lon_grid:
    sphere_line = torch.tensor([[lat, lon] for lat in lat_grid])
    plane_line = sphere2plane(sphere_line)
    ax2.plot(plane_line[:, 1].numpy(), plane_line[:, 0].numpy(), 'b-', alpha=0.3, linewidth=0.5)
ax2.set_xlabel('X (经度方向)')
ax2.set_ylabel('Y (纬度方向)')
ax2.set_title('等距柱状投影网格')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# 子图3: 关键点的转换路径
ax3 = fig.add_subplot(2, 3, 3)
key_points = torch.tensor([[0, 0], [90, 0], [-90, 0], [0, 90], [0, -90], [45, 45]])
key_plane = sphere2plane(key_points)
ax3.scatter(key_plane[:, 1].numpy(), key_plane[:, 0].numpy(), c='red', s=100, zorder=5)
for i, (sp, pl) in enumerate(zip(key_points, key_plane)):
    ax3.annotate(f'({sp[0]:.0f}°,{sp[1]:.0f}°)',
                 (pl[1].item(), pl[0].item()),
                 xytext=(5, 5), textcoords='offset points', fontsize=8)
ax3.set_xlabel('X (经度方向)')
ax3.set_ylabel('Y (纬度方向)')
ax3.set_title('关键点的平面投影')
ax3.set_xlim(-0.1, 1.1)
ax3.set_ylim(-0.1, 1.1)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

# 子图4: 纬度对Y坐标的影响
ax4 = fig.add_subplot(2, 3, 4)
lat_range = torch.linspace(-90, 90, 181)
lon_fixed = torch.zeros_like(lat_range)
sphere_test = torch.stack([lat_range, lon_fixed], dim=1)
plane_test = sphere2plane(sphere_test)
ax4.plot(lat_range.numpy(), plane_test[:, 0].numpy(), 'b-', linewidth=2)
ax4.set_xlabel('纬度 (度)')
ax4.set_ylabel('归一化Y坐标')
ax4.set_title('纬度 -> Y坐标映射')
ax4.grid(True, alpha=0.3)

# 子图5: 经度对X坐标的影响
ax5 = fig.add_subplot(2, 3, 5)
lon_range = torch.linspace(-180, 180, 361)
lat_fixed = torch.zeros_like(lon_range)
sphere_test2 = torch.stack([lat_fixed, lon_range], dim=1)
plane_test2 = sphere2plane(sphere_test2)
ax5.plot(lon_range.numpy(), plane_test2[:, 1].numpy(), 'r-', linewidth=2)
ax5.set_xlabel('经度 (度)')
ax5.set_ylabel('归一化X坐标')
ax5.set_title('经度 -> X坐标映射')
ax5.grid(True, alpha=0.3)

# 子图6: 转换路径示例
ax6 = fig.add_subplot(2, 3, 6)
example_sphere = torch.tensor([[30.0, 45.0]])
example_xyz = sphere2xyz(example_sphere)
example_plane = sphere2plane(example_sphere)
example_plane_from_xyz = xyz2plane(example_xyz)

ax6.text(0.1, 0.9, f'球面坐标: ({example_sphere[0, 0]:.1f}°, {example_sphere[0, 1]:.1f}°)',
         transform=ax6.transAxes, fontsize=10, verticalalignment='top')
ax6.text(0.1, 0.75, f'3D坐标: ({example_xyz[0, 0]:.3f}, {example_xyz[0, 1]:.3f}, {example_xyz[0, 2]:.3f})',
         transform=ax6.transAxes, fontsize=10, verticalalignment='top')
ax6.text(0.1, 0.6, f'平面坐标: (y={example_plane[0, 0]:.3f}, x={example_plane[0, 1]:.3f})',
         transform=ax6.transAxes, fontsize=10, verticalalignment='top')
ax6.text(0.1, 0.45, f'验证: 从xyz转平面 = ({example_plane_from_xyz[0, 0]:.3f}, {example_plane_from_xyz[0, 1]:.3f})',
         transform=ax6.transAxes, fontsize=10, verticalalignment='top')
ax6.axis('off')
ax6.set_title('转换示例')

plt.tight_layout()

# 确定保存路径 - 保存到当前脚本所在目录
try:
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 如果__file__不存在（如在交互式环境中），使用当前工作目录
    script_dir = os.getcwd()

save_path = os.path.join(script_dir, '坐标转换可视化.png')

# 确保目录存在
os.makedirs(script_dir, exist_ok=True)

plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"✓ 可视化图像已保存到: {save_path}")

# ============================================================================
# 第八部分：实际应用示例
# ============================================================================

print("\n" + "=" * 80)
print("第八部分：实际应用示例")
print("=" * 80)

print("\n示例1: 眼动追踪数据转换")
print("-" * 80)
# 模拟眼动追踪数据（球面坐标）
gaze_sphere = torch.tensor([
    [10.5, 45.2],  # 第1个注视点
    [12.3, 46.8],  # 第2个注视点
    [15.1, 48.5],  # 第3个注视点
])
print("原始眼动数据（球面坐标）:")
for i, g in enumerate(gaze_sphere):
    print(f"  注视点{i + 1}: ({g[0]:.1f}°, {g[1]:.1f}°)")

# 转换为3D坐标（用于模型处理）
gaze_xyz = sphere2xyz(gaze_sphere)
print("\n转换为3D坐标（用于模型）:")
for i, g in enumerate(gaze_xyz):
    print(f"  注视点{i + 1}: ({g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f})")

# 转换为平面坐标（用于可视化）
gaze_plane = sphere2plane(gaze_sphere, (128, 256))
print("\n转换为平面坐标（用于可视化，图像尺寸128x256）:")
for i, g in enumerate(gaze_plane):
    print(f"  注视点{i + 1}: 像素位置 (y={g[0]:.1f}, x={g[1]:.1f})")

print("\n示例2: 图像坐标到3D坐标的转换")
print("-" * 80)
# 假设用户在图像上的某个位置点击
image_height, image_width = 128, 256
click_y, click_x = 64, 128  # 图像中心
print(f"用户在图像上点击: 像素位置 (y={click_y}, x={click_x})")

# 转换为归一化坐标
normalized_y = click_y / image_height
normalized_x = click_x / image_width
print(f"归一化坐标: (y={normalized_y:.3f}, x={normalized_x:.3f})")

# 转换为球面坐标
click_plane = torch.tensor([[normalized_y, normalized_x]])
click_sphere = plane2sphere(click_plane, (image_height, image_width))
print(f"球面坐标: ({click_sphere[0, 0]:.2f}°, {click_sphere[0, 1]:.2f}°)")

# 转换为3D坐标
click_xyz = sphere2xyz(click_sphere)
print(f"3D坐标: ({click_xyz[0, 0]:.4f}, {click_xyz[0, 1]:.4f}, {click_xyz[0, 2]:.4f})")

print("\n" + "=" * 80)
print("代码示例运行完成！")
print("=" * 80)
print("\n💡 学习建议:")
print("1. 尝试修改测试点，观察转换结果")
print("2. 理解为什么需要这些不同的坐标系统")
print("3. 思考在眼动追踪中，为什么使用3D坐标而不是2D坐标")
print("4. 理解等距柱状投影的特点和局限性")
