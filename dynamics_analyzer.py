import os
import sys
import glob
import re
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy
from matplotlib.font_manager import fontManager

# ==============================================================================
#  字体设置 (最终解决方案)
# ==============================================================================

def find_and_set_chinese_font():
    """
    动态查找并设置系统上可用的中文字体。
    """
    # 优先尝试的常见中文字体列表
    font_preferences = [
        'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti SC', 
        'WenQuanYi Micro Hei', 'sans-serif'
    ]
    
    # 获取系统上所有matplotlib可用的字体名称
    available_fonts = set(f.name for f in fontManager.ttflist)
    
    found_font = None
    for font in font_preferences:
        if font in available_fonts:
            found_font = font
            break
            
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font]
        print(f"字体信息：已自动选择并设置中文字体 '{found_font}'。")
    else:
        print("警告：未在系统中找到推荐的中文字体。图表中的中文可能无法正常显示。")
        # 降级到任何可用的无衬线字体
        plt.rcParams['font.sans-serif'] = ['sans-serif']

    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

# 在程序开始时调用字体设置函数
find_and_set_chinese_font()


# ==============================================================================
#  HELPER FUNCTIONS (Adapted from GUI)
# ==============================================================================

def load_fragment_data(experiment_path):
    """加载一个实验的所有 fragment properties 数据。"""
    csv_dir = os.path.join(experiment_path, 'csv')
    if not os.path.isdir(csv_dir):
        print(f"错误：在 '{experiment_path}' 中未找到 'csv' 文件夹。")
        return None
    csv_files = glob.glob(os.path.join(csv_dir, 'fragments_properties_step_*.csv'))
    if not csv_files:
        print(f"错误：在 '{csv_dir}' 中未找到 'fragments_properties_step_*.csv' 文件。")
        return None
    
    all_fragments_df = []
    try:
        csv_files.sort(key=lambda f: int(re.search(r'step_(\d+)\.csv', f).group(1)))
    except AttributeError:
        print("错误：无法从文件名中解析step编号。")
        return None

    for f in csv_files:
        try:
            match = re.search(r'step_(\d+)\.csv', os.path.basename(f))
            if match:
                step = int(match.group(1))
                df = pd.read_csv(f)
                df['step'] = step
                all_fragments_df.append(df)
        except Exception as e:
            print(f"警告：读取文件 {f} 时出错: {e}")
            continue
            
    if not all_fragments_df:
        print("错误：未能成功加载任何 fragment 数据。")
        return None
        
    full_df = pd.concat(all_fragments_df, ignore_index=True)
    full_df['parent_id'] = full_df['parent_id'].astype(int)
    if 'fragment_id' in full_df.columns:
        full_df['fragment_id'] = full_df['fragment_id'].astype(int)
    return full_df

def build_genealogy_graph(df):
    """构建完整的谱系图。"""
    G = nx.DiGraph()
    for _, row in df.iterrows():
        node_id = (row['step'], row['fragment_id'])
        G.add_node(node_id, area=row['area'], num_balls=row['num_balls'])
    
    for _, row in df.iterrows():
        if row['parent_id'] != -1:
            parent_node_id = (row['step'] - 1, row['parent_id'])
            child_node_id = (row['step'], row['fragment_id'])
            if parent_node_id in G:
                G.add_edge(parent_node_id, child_node_id)
    return G

# ==============================================================================
#  METRIC CALCULATION FUNCTIONS
# ==============================================================================

def calculate_metrics_for_step(step, full_df, G):
    """为单个时间步计算所有度量指标。"""
    
    # 筛选当前和上一时刻的数据
    df_current = full_df[full_df['step'] == step]
    df_previous = full_df[full_df['step'] == step - 1]
    
    if df_current.empty:
        return {}

    metrics = {'step': step}
    
    # --- 分量一：碎裂度指数 (FI) ---
    initial_block_count = len(full_df[full_df['step'] == full_df['step'].min()])
    metrics['FI_N_norm'] = len(df_current) / initial_block_count if initial_block_count > 0 else 0
    
    block_sizes = df_current['area']
    if len(block_sizes) > 1:
        size_distribution = block_sizes.value_counts(normalize=True)
        metrics['FI_S_entropy'] = entropy(size_distribution, base=2)
    else:
        metrics['FI_S_entropy'] = 0

    # --- 分量二：演化强度指数 (EI) ---
    metrics['EI_R_new'] = (len(df_current) - len(df_previous)) if not df_previous.empty else len(df_current)

    # 优势家族增长率 G_dom
    subgraph_current = G.subgraph([n for n in G.nodes() if n[0] <= step])
    families_current = [c for c in nx.weakly_connected_components(subgraph_current)]
    max_family_size_current = max(len(f) for f in families_current) if families_current else 0
    
    if not df_previous.empty:
        subgraph_previous = G.subgraph([n for n in G.nodes() if n[0] <= step - 1])
        families_previous = [c for c in nx.weakly_connected_components(subgraph_previous)]
        max_family_size_previous = max(len(f) for f in families_previous) if families_previous else 0
    else:
        max_family_size_previous = 0
    metrics['EI_G_dom'] = max_family_size_current - max_family_size_previous

    # --- 分量三：结构复杂性指数 (SCI) ---
    nodes_in_subgraph = list(subgraph_current.nodes())
    branching_factors = [subgraph_current.out_degree(n) for n in nodes_in_subgraph if subgraph_current.out_degree(n) > 0]
    metrics['SCI_B_bar'] = np.mean(branching_factors) if branching_factors else 0
    
    # M_topo: 简化版 - 计算分支因子大于2的节点比例
    complex_branches = [bf for bf in branching_factors if bf > 2]
    metrics['SCI_M_topo_proxy'] = len(complex_branches) / len(nodes_in_subgraph) if len(nodes_in_subgraph) > 0 else 0
    
    return metrics

# ==============================================================================
#  MAIN ANALYSIS AND PLOTTING
# ==============================================================================

def main(experiment_path):
    """主函数，执行整个分析流程。"""
    print(f"--- 开始分析实验: {os.path.basename(experiment_path)} ---")
    
    # 1. 加载数据和构建图
    print("1/4: 正在加载所有碎片数据...")
    full_df = load_fragment_data(experiment_path)
    if full_df is None:
        return
    print(f"    加载了 {len(full_df)} 条记录，覆盖 {full_df['step'].nunique()} 个时间步。")

    print("2/4: 正在构建完整谱系图...")
    G = build_genealogy_graph(full_df)
    print(f"    谱系图构建完成，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")

    # 2. 逐时间步计算指标
    print("3/4: 正在逐时间步计算动力学指标...")
    all_steps = sorted(full_df['step'].unique())
    results = []
    for step in all_steps:
        step_metrics = calculate_metrics_for_step(step, full_df, G)
        if step_metrics:
            results.append(step_metrics)
        print(f"\r    已完成 Step {step}/{all_steps[-1]}", end="")
    print("\n    指标计算完成。")

    results_df = pd.DataFrame(results)
    
    # 3. 保存结果到CSV
    output_dir = os.path.join(experiment_path, 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'dynamics_metrics.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"    分析结果已保存到: {csv_path}")

    # 4. 绘制并保存图表
    print("4/4: 正在生成并保存结果图表...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f'岩层破断动力学指标演化\n实验: {os.path.basename(experiment_path)}', fontsize=16)
    plt.style.use('seaborn-v0_8-whitegrid')

    # 绘图
    results_df.plot(x='step', y='FI_N_norm', ax=axes[0, 0], title='FI: 标准化块体数量 (N_norm)')
    results_df.plot(x='step', y='FI_S_entropy', ax=axes[0, 1], title='FI: 熵权尺寸 (S_entropy)')
    results_df.plot(x='step', y='EI_R_new', ax=axes[1, 0], title='EI: 新生块体率 (R_new)')
    results_df.plot(x='step', y='EI_G_dom', ax=axes[1, 1], title='EI: 优势家族增长率 (G_dom)')
    results_df.plot(x='step', y='SCI_B_bar', ax=axes[2, 0], title='SCI: 系统平均分支因子 (B_bar)')
    results_df.plot(x='step', y='SCI_M_topo_proxy', ax=axes[2, 1], title='SCI: 拓扑模式指标代理 (M_topo_proxy)')

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("时间步 (Step)")
            ax.legend(loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(output_dir, 'dynamics_plot.png')
    plt.savefig(plot_path)
    print(f"    结果图表已保存到: {plot_path}")
    plt.close()
    
    print("--- 分析完成 ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python dynamics_analyzer.py <path_to_experiment_folder>")
    else:
        experiment_path = sys.argv[1]
        main(experiment_path)
