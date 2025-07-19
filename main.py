import csv
import itasca
from itasca import ball, wall
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import *

# ==============================================================================
# 0. 全局设置
# ==============================================================================
deterministic_mode = True # 设为True可保证每次运行结果一致

# ==============================================================================
# 1. 定义岩层地质 (单位: 米)
# ==============================================================================
# 直接定义各岩层厚度 (单位: 米)，顺序与您提供的表格一致 (从上到下)
thicknesses_in_meters = [
    0.2050, 0.0300, 0.0750, 0.3000, 0.0800, 0.0750, 0.0650, 0.0600, 0.0450, 
    0.0700, 0.0900, 0.0650, 0.0650, 0.0750, 0.0200, 0.0900, 0.0600, 0.0300, 
    0.0350, 0.0550
]

# *** 修正：颠倒岩层列表 ***
# 您的表格是从上到下，而建模需要从下到上。因此，将列表顺序反转。
thicknesses_in_meters.reverse()

# 自动计算累积高度，这将是新的 layer_array
# 这个数组定义了从下到上每个地层的顶部高程 (单位: 米)
cumulative_heights = []
current_height = 0
for thickness in thicknesses_in_meters:
    current_height += thickness
    cumulative_heights.append(round(current_height, 4))

layer_array = cumulative_heights
# 根据您的数据，模型总高度约为 1.6m, 模型宽度应在.dat文件中设为 2.5m

# ==============================================================================
# 2. 设置模型和模拟参数 (单位: 米)
# ==============================================================================

# --- 开挖参数 (单位: 米) ---
# 注意：模型总宽2.5m，请确保开挖参数合理
subsurface_level = 0.1   # 地表土层厚度 (m)

# 根据用户要求，从距左侧边界45cm处开始开挖，到距右侧45cm处停止
left_pillar_width = 0.45 # 左侧煤柱宽度 (m)
right_pillar_width = 0.45 # 右侧煤柱宽度 (m)

# 将左侧煤柱宽度设为第一个分区的长度
first_section_length = left_pillar_width

# 定义每个开挖步的宽度
sec_interval = 0.2   # 后续开挖区块长度 (m)

# --- 力学参数 (重要提示!) ---
# 以下参数为示例值，并非针对您的特定岩层。
# 真实模拟需要对每一种岩性（如中粒砂岩、粉砂岩等）进行参数标定，
# 以获得与实际岩石力学特性匹配的微观参数集。
fric = 0.05    # 摩擦系数 (无单位)
rfric = 0.0   # 滚动摩擦系数 (无单位)
dpnr = 0.2   # 法向阻尼系数 (无单位)
dpsr = 0.2    # 切向阻尼系数 (无单位)

# --- 故障排除：模型收缩问题的参数调整 ---
# 您遇到的模型“缩成一块”的问题，是由于颗粒刚度(stiffness)与吸引力(cohesion)不匹配导致的。
# 当吸引力 F0 很大，而颗粒刚度 emod 太小时，颗粒就会被过度“压缩”，导致模型坍缩。
#
# 解决方案:
# 1. 打开您的 .dat 文件 (很可能是 yuya-new.dat 或 pingheng-linear.dat)。
# 2. 找到定义颗粒刚度的命令，通常是 "ball attribute emod ..." 或在 "ball contact-model" 命令中。
# 3. 大幅提高 emod 的值。例如，如果当前是 emod 1e6，请尝试修改为 emod 1e7 或 emod 1e8。
# 4. 保存 .dat 文件并重新运行此 Python 脚本。
#
# 提高刚度可以使颗粒有效抵抗强大的吸引力，从而在保持粘聚力的同时维持模型的稳定。

F0 = 1e5    # 地表层最大吸引力 (N) -> 已恢复原值
D0 = 0.2   # 地表层吸引力范围 (m)

# --- 模拟流程控制 ---
# *** 修正：更新开挖煤层编号 ***
# 由于岩层顺序已颠倒，原先从顶部数第10层的"3-1 coal"，现在是从底部数第11层。
# utils.py中的fenceng函数从'1'开始命名地层组。
excavation_layer_group = '11' # 开挖第11层: 3-1 coal。

# 自动计算开挖的起始与终止分区
# 第0区是左侧煤柱，因此从第1区开始开挖
excavation_start_section = 1

# 计算需要开挖多少个分区
model_width = 2.5 # 模型总宽度
excavation_width = model_width - left_pillar_width - right_pillar_width
num_excavation_sections = int(round(excavation_width / sec_interval))

# 计算终止开挖的分区编号
excavation_end_section = excavation_start_section + num_excavation_sections - 1

step_solve_time = 3  # 定义每步开挖后的求解时间 (秒)

def run_simulation(**params):

    # 创建结果路径
    resu_path = f'experiments/exp_{fric}_{rfric}_{dpnr}_{dpsr}_{F0}_{D0}'
    
    # 创建结果文件夹
    if not os.path.exists(resu_path):
        os.makedirs(resu_path)
        os.makedirs(os.path.join(resu_path, 'img'))
        os.makedirs(os.path.join(resu_path, 'sav')) 
        os.makedirs(os.path.join(resu_path, 'mat'))

    # 防止PFC重置Python状态
    itasca.command("python-reset-state false")

    itasca.command("model new")

    itasca.set_deterministic(deterministic_mode)

    # 1. 生成初始颗粒模型 (yuya-new.dat)
    # 重要: 请确保 yuya-new.dat 文件中的墙体宽度为2.5m，颗粒半径为0.005-0.0075m
    run_dat_file("yuya-new.dat")
    itasca.command("model save 'yuya'")

    # 删除墙体外的颗粒
    delete_balls_outside_area(
        x_min=wall.find('boxWallLeft4').pos_x(),
        x_max=wall.find('boxWallRight2').pos_x(),
        y_min=wall.find('boxWallBottom1').pos_y(),
        y_max=wall.find('boxWallTop3').pos_y()
    )

    # 2. 对模型进行分层与分区
    itasca.command("model restore 'yuya'")
    sec_num = fenceng(
        sec_interval=sec_interval, 
        layer_array=layer_array,
        first_section_length=first_section_length,
        subsurface_level=subsurface_level
    )
    itasca.command("model save 'fenceng'")

    # 3. 初始平衡
    itasca.command("model restore 'fenceng'")

    wall_up_pos_y = wall.find('boxWallTop3').pos_y()
    wlx, wly = compute_dimensions()

    # 将力学参数传递给PFC
    itasca.fish.set('fric', fric)
    itasca.fish.set('rfric', rfric)
    itasca.fish.set('dpnr', dpnr)
    itasca.fish.set('dpsr', dpsr)
    itasca.fish.set('F0', F0)
    itasca.fish.set('D0', D0)

    # 运行平衡计算脚本
    run_dat_file("pingheng-linear.dat")
    itasca.command(f"model save '{os.path.join(resu_path, 'sav', 'pingheng')}'")

    # 4. 模拟开挖
    itasca.command(f"model restore '{os.path.join(resu_path, 'sav', 'pingheng')}'")
    
    # 重置颗粒位移等属性
    itasca.command("ball attribute velocity 0 spin 0 displacement 0")

    # 获取模型顶部用于监测地表沉降的颗粒
    top_ball_pos = get_balls_max_pos(1)

    # 检查模型是否发生过度膨胀或收缩
    if top_ball_pos > wall_up_pos_y * 1.1:
        print(f"警告: 模型正在膨胀，请检查参数。当前顶部高度: {top_ball_pos}, 模型墙高: {wall_up_pos_y}")
    if top_ball_pos < wall_up_pos_y * 0.8:
        print(f"警告: 模型正在收缩，请检查参数。当前顶部高度: {top_ball_pos}, 模型墙高: {wall_up_pos_y}")

    rdmax = itasca.fish.get('rdmax')
    
    # 将每个分区的顶部颗粒存入字典，用于后续位移监测
    ball_objects_dict = {}
    for i in range(0, sec_num):
        ball_objects_dict[str(i)] = get_balls_object_in_area(str(i), top_ball_pos-rdmax*1.5, top_ball_pos)

    # 删除没有监测到颗粒的空分区
    empty_sections = [k for k, v in ball_objects_dict.items() if not v]
    if empty_sections:
        print(f"警告: 以下分区未找到监测颗粒，将被忽略: {empty_sections}")
        ball_objects_dict = {k: v for k, v in ball_objects_dict.items() if v}
        sec_num = len(ball_objects_dict)

    y_disps_list = {}
    
    # 开始循环开挖 (使用新的起止分区)
    for i in range(excavation_start_section, excavation_end_section + 1):
        # 'excavation_pos' 代表当前开挖工作面的x坐标
        excavation_pos = first_section_length + (i - excavation_start_section + 1) * sec_interval
        
        # 删除指定煤层和分区内的颗粒
        for ball_obj in list(ball.list()):
            if ball_obj.valid():
                if ball_obj.in_group(excavation_layer_group, 'layer') and ball_obj.in_group(f'{i}', 'section'):
                    try:
                        ball_obj.delete()
                    except Exception as e:
                        print(f"删除颗粒 {ball_obj.id()} 时出错: {str(e)}")
        
        # 求解计算
        itasca.command(f"model solve cycle 10")
        global_timestep = itasca.timestep()
        step_interval_cycles = int(step_solve_time / global_timestep)
        if i < excavation_start_section + 2: # 在开挖初期使用更严格的平衡标准
            itasca.command(f"model solve cycle {step_interval_cycles} or ratio-average 1e-5")
        else:
            itasca.command(f"model solve cycle {step_interval_cycles} or ratio-average 1e-3")
        itasca.command(f"model save '{os.path.join(resu_path, 'sav', str(i))}'")

        # 获取每个分区的平均垂直位移并记录
        y_disps = [get_avg_ball_y_disp(ball_objects_dict[str(k)]) for k in range(0, sec_num)]
        y_disps_list[excavation_pos] = y_disps

        # 绘制并保存当前步骤的位移云图
        plot_y_displacement_heatmap(window_size=rdmax * 2, model_width=wlx, model_height=wly, name=f"{excavation_pos:.2f}", interpolate='nearest', resu_path=resu_path)

    # 5. 结果输出
    # 绘制所有开挖步骤的地表沉降曲线
    plt.figure(figsize=(10, 6))
    for excavation_pos, y_disps in y_disps_list.items():
        x_positions = [first_section_length / 2 if first_section_length > 0 else sec_interval / 2] + \
                    [first_section_length + k * sec_interval + sec_interval / 2 if first_section_length > 0 else k * sec_interval + sec_interval / 2 for k in range(0, sec_num - 2)] + \
                    [int(wlx)]
        plt.plot(x_positions, y_disps, label=f'开挖至 {excavation_pos:.2f} m')
    plt.xlabel('工作面位置 (m)')
    plt.xlim(0, int(wlx))
    plt.ylabel('垂直位移 (m)')
    plt.title('地表沉降曲线')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='small')
    plt.savefig(os.path.join(resu_path, 'img', 'surface_y_disp_vs_section.png'), dpi=400, bbox_inches='tight')
    plt.close()

    # 将沉降数据保存到CSV文件
    with open(os.path.join(resu_path, 'surface_y_disp_vs_section.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['监测点位置(m)'] + list(np.fromiter(y_disps_list.keys(), dtype=float)))
        for section in range(0, sec_num):
            if section == 0:
                x_position = first_section_length / 2 if first_section_length > 0 else sec_interval / 2
            elif section == sec_num - 1:
                x_position = int(wlx)
            else:
                x_position = first_section_length + (section - 1) * sec_interval + sec_interval / 2 if first_section_length > 0 else (section - 1) * sec_interval + sec_interval / 2
            row = [x_position] + [y_disps_list[step][section] for step in y_disps_list]
            writer.writerow(row)
    
    print(f"模拟完成，结果已保存至: {resu_path}")
    return True
    
if __name__ == "__main__":
    run_simulation(
        deterministic_mode=deterministic_mode,
        fric=fric, 
        rfric=rfric, 
        dpnr=dpnr, 
        dpsr=dpsr, 
        F0=F0,
        D0=D0,
        step_solve_time=step_solve_time,
        layer_array=layer_array,
        first_section_length=first_section_length,
        subsurface_level=subsurface_level,
        sec_interval=sec_interval,
        excavation_layer_group=excavation_layer_group
    )
