import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

# ================= 配置参数 =================
N_SAMPLES = 8       # 样本数量
FIG_SIZE = (8, 8)   # 图片大小
COLOR_3D = '#FF8C00'  # 3D点颜色 (您的主题橙色)
COLOR_PROJ = '#1E90FF' # 投影点颜色 (您的主题蓝色)
# COLOR_GRID = '#BBBBBB' # 网格线颜色 (已不再需要)
COLOR_CONNECT = '#FF8C00' # 连接线颜色
FONT_SIZE_LABEL = 14
DPI = 300

# ================= 数据生成 (LHS) =================
sampler = qmc.LatinHypercube(d=3, seed=42) # 固定种子以复现结果
lhs_samples = sampler.random(n=N_SAMPLES)

# ================= 3D 绘图主程序 =================
fig = plt.figure(figsize=FIG_SIZE)
ax = fig.add_subplot(111, projection='3d')

x, y, z = lhs_samples[:, 0], lhs_samples[:, 1], lhs_samples[:, 2]

# --- 1. 设置3D空间和视角 ---
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_box_aspect((1, 1, 1)) # 正方体
ax.view_init(elev=25, azim=-55) # 视角

# --- 2. 绘制立方体墙面上的网格线 (已注释掉) ---
#这部分代码被注释掉，以移除繁杂的网格线，使图表更清爽。
# ticks = np.linspace(0, 1, N_SAMPLES + 1)
# for t in ticks:
#     # XY平面 (底面 z=0)
#     ax.plot([t, t], [0, 1], [0, 0], color=COLOR_GRID, lw=1, ls='-')
#     ax.plot([0, 1], [t, t], [0, 0], color=COLOR_GRID, lw=1, ls='-')
#     # XZ平面 (背面 y=1)
#     ax.plot([t, t], [1, 1], [0, 1], color=COLOR_GRID, lw=1, ls='-')
#     ax.plot([0, 1], [1, 1], [t, t], color=COLOR_GRID, lw=1, ls='-')
#     # YZ平面 (侧面 x=0)
#     ax.plot([0, 0], [t, t], [0, 1], color=COLOR_GRID, lw=1, ls='-')
#     ax.plot([0, 0], [0, 1], [t, t], color=COLOR_GRID, lw=1, ls='-')

# --- 3. 绘制投影点 (Projections) ---
# 使用蓝色，暗示它们是"静态参数"的投影
proj_size = 60
proj_alpha = 0.7
ax.scatter(x, y, np.zeros_like(z), c=COLOR_PROJ, s=proj_size, alpha=proj_alpha, marker='o', zorder=5) # XY
ax.scatter(x, np.ones_like(y), z, c=COLOR_PROJ, s=proj_size, alpha=proj_alpha, marker='o', zorder=5) # XZ
ax.scatter(np.zeros_like(x), y, z, c=COLOR_PROJ, s=proj_size, alpha=proj_alpha, marker='o', zorder=5) # YZ

# --- 4. 绘制连接线 (虚线) ---
# 连接3D点和它的三个投影，增强空间感
for i in range(N_SAMPLES):
    ax.plot([x[i], x[i]], [y[i], y[i]], [0, z[i]], color=COLOR_CONNECT, ls='--', lw=1, alpha=0.6) # to XY
    ax.plot([x[i], x[i]], [y[i], 1], [z[i], z[i]], color=COLOR_CONNECT, ls='--', lw=1, alpha=0.6) # to XZ
    ax.plot([0, x[i]], [y[i], y[i]], [z[i], z[i]], color=COLOR_CONNECT, ls='--', lw=1, alpha=0.6) # to YZ

# --- 5. 绘制空间中的真实 3D 点 ---
# 使用橙色，作为最显眼的主体
ax.scatter(x, y, z, c=COLOR_3D, s=180, edgecolor='white', linewidth=1.5, zorder=100)

# --- 6. 美化与标注 ---
# 隐藏刻度数字
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# 设置坐标轴标签
# ax.set_xlabel('\nParameter 1\n(e.g., Elastic Modulus)', fontsize=FONT_SIZE_LABEL, labelpad=15)
# ax.set_ylabel('\nParameter 2\n(e.g., Tensile Strength)', fontsize=FONT_SIZE_LABEL, labelpad=15)
# ax.set_zlabel('\nParameter 3\n(e.g., Layer Thickness)', fontsize=FONT_SIZE_LABEL, labelpad=15)

# 移除背景色，使其透明
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

plt.tight_layout()

# 保存图片
plt.savefig('LHS_Single_Schematic_NoGrid.pdf', dpi=DPI, bbox_inches='tight', transparent=True)
plt.savefig('LHS_Single_Schematic_NoGrid.png', dpi=DPI, bbox_inches='tight', transparent=True)

# 显示
plt.show()