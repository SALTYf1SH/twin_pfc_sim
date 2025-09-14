import tkinter as tk
from tkinter import ttk, messagebox
import os
import glob
import pandas as pd
import networkx as nx
import re
import math
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import fontManager

# ==============================================================================
#  字体设置 (最终解决方案)
# ==============================================================================

def find_and_set_chinese_font():
    """
    动态查找并设置系统上可用的中文字体。
    """
    font_preferences = [
        'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti SC', 
        'WenQuanYi Micro Hei', 'sans-serif'
    ]
    available_fonts = set(f.name for f in fontManager.ttflist)
    found_font = None
    for font in font_preferences:
        if font in available_fonts:
            found_font = font
            break
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font]
    else:
        plt.rcParams['font.sans-serif'] = ['sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

find_and_set_chinese_font()

# ==============================================================================
#  ANALYSIS LOGIC
# ==============================================================================

def load_fragment_data(experiment_path):
    csv_dir = os.path.join(experiment_path, 'csv')
    csv_files = glob.glob(os.path.join(csv_dir, 'fragments_properties_step_*.csv'))
    if not csv_files: return None
    all_fragments_df = []
    try:
        csv_files.sort(key=lambda f: int(re.search(r'step_(\d+)\.csv', f).group(1)))
    except AttributeError: return None
    for f in csv_files:
        try:
            match = re.search(r'step_(\d+)\.csv', os.path.basename(f))
            if match:
                step = int(match.group(1))
                df = pd.read_csv(f)
                df['step'] = step
                all_fragments_df.append(df)
        except Exception: continue
    if not all_fragments_df: return None
    full_df = pd.concat(all_fragments_df, ignore_index=True)
    full_df['parent_id'] = full_df['parent_id'].astype(int)
    if 'fragment_id' in full_df.columns:
        full_df['fragment_id'] = full_df['fragment_id'].astype(int)
    return full_df

def load_ball_data_for_step(experiment_path, step):
    csv_path = os.path.join(experiment_path, 'csv', f'fragments_balls_step_{step}.csv')
    if not os.path.exists(csv_path): return None
    try:
        df = pd.read_csv(csv_path)
        expected_cols = ['x', 'y', 'radius', 'fragment_id']
        if not all(col in df.columns for col in expected_cols):
            df = pd.read_csv(csv_path, header=None, names=expected_cols, skiprows=1)
            if not all(col in df.columns for col in expected_cols): return None
        return df
    except Exception as e:
        print(f"Error loading ball data for step {step}: {e}")
        return None

def build_genealogy_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        node_id = (row['step'], row['fragment_id'])
        G.add_node(node_id, area=row['area'], num_balls=row['num_balls'], centroid_x=row['centroid_x'], centroid_y=row['centroid_y'])
    for _, row in df.iterrows():
        if row['parent_id'] != -1:
            parent_node_id = (row['step'] - 1, row['parent_id'])
            child_node_id = (row['step'], row['fragment_id'])
            if parent_node_id in G: G.add_edge(parent_node_id, child_node_id)
    return G

def plot_family_tree_with_shapes(full_graph, start_node, output_filename, experiment_path):
    """ 
    Renders the family tree with actual fragment shapes as nodes.
    DEFINITIVE FIX: Ensures absolute, forward-slash paths are used for Graphviz.
    """
    try:
        import graphviz
    except ImportError: return False, "Graphviz library not found. Please run: pip install graphviz"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir_path = os.path.join(script_dir, "_temp_node_images")
    if os.path.exists(temp_dir_path): shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)

    ancestors = nx.ancestors(full_graph, start_node)
    descendants = nx.descendants(full_graph, start_node)
    family_nodes = ancestors.union(descendants).union({start_node})
    
    node_image_paths = {}
    print("Rendering family member shapes...")
    for i, node in enumerate(family_nodes):
        # --- FIX 1: Ensure node identifiers are clean integers ---
        step, frag_id = int(node[0]), int(node[1])
        
        print(f"  -> Rendering node {i+1}/{len(family_nodes)}: Step {step}, Fragment {frag_id}")
        ball_df = load_ball_data_for_step(experiment_path, step)
        if ball_df is None or ball_df.empty: continue
        frag_balls = ball_df[ball_df['fragment_id'] == frag_id]
        if frag_balls.empty: continue

        fig = Figure(figsize=(1.5, 1.5), dpi=80)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        
        node_color = 'dimgray'
        for _, ball in frag_balls.iterrows():
            circle = plt.Circle((ball['x'], ball['y']), radius=ball['radius'], color=node_color)
            ax.add_patch(circle)
        
        x_min_ball = (frag_balls['x'] - frag_balls['radius']).min()
        x_max_ball = (frag_balls['x'] + frag_balls['radius']).max()
        y_min_ball = (frag_balls['y'] - frag_balls['radius']).min()
        y_max_ball = (frag_balls['y'] + frag_balls['radius']).max()
        center_x = (x_min_ball + x_max_ball) / 2
        center_y = (y_min_ball + y_max_ball) / 2
        max_range = max(x_max_ball - x_min_ball, y_max_ball - y_min_ball)
        if max_range == 0: max_range = frag_balls['radius'].iloc[0] * 2.2
        else: max_range *= 1.1
        ax.set_xlim(center_x - max_range / 2, center_x + max_range / 2)
        ax.set_ylim(center_y - max_range / 2, center_y + max_range / 2)
        
        ax.set_axis_off()
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        img_path = os.path.join(temp_dir_path, f"S{step}_F{frag_id}.png")
        fig.savefig(img_path, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        
        # --- FIX 2: Create a normalized, absolute path for Graphviz ---
        absolute_img_path = os.path.abspath(img_path)
        normalized_path = absolute_img_path.replace('\\', '/')
        node_image_paths[node] = normalized_path

    print("Assembling final family tree...")
    dot = graphviz.Digraph(comment=f'Family Tree of {start_node}')
    dot.attr(rankdir='TB', splines='spline', overlap='false')
    family_graph = full_graph.subgraph(family_nodes)

    for node in family_graph.nodes():
        if node not in node_image_paths: continue
        node_props = full_graph.nodes[node]
        label_text = f"S{int(node[0])} F{int(node[1])}\nArea: {node_props.get('area', 0):.1f}"
        dot.node(name=str(node), label=label_text, image=node_image_paths[node], shape='none', fontcolor='black', fontsize='10')

    for edge in family_graph.edges: dot.edge(str(edge[0]), str(edge[1]))

    try:
        output_format = os.path.splitext(output_filename)[1][1:] or 'png'
        dot.render(os.path.splitext(output_filename)[0], format=output_format, view=False, cleanup=True)
        shutil.rmtree(temp_dir_path)
        return True, f"Success! Image saved to '{output_filename}'"
    except graphviz.backend.ExecutableNotFound:
        return False, "ERROR: Graphviz executable not found. Please install it and add to PATH."
    except Exception as e:
        if os.path.exists(temp_dir_path): shutil.rmtree(temp_dir_path)
        return False, f"An error occurred during rendering: {e}"

def plot_family_tree_with_variable_thickness(full_graph, start_node, output_filename):
    """ 
    Renders the family tree with edge thickness proportional to num_balls.
    """
    try:
        import graphviz
    except ImportError: return False, "Graphviz library not found. Please run: pip install graphviz"

    ancestors = nx.ancestors(full_graph, start_node)
    descendants = nx.descendants(full_graph, start_node)
    family_nodes = ancestors.union(descendants).union({start_node})
    family_graph = full_graph.subgraph(family_nodes)

    ball_counts = [full_graph.nodes[n].get('num_balls', 1) for n in family_nodes]
    min_balls, max_balls = min(ball_counts) if ball_counts else 1, max(ball_counts) if ball_counts else 1
    
    def map_balls_to_thickness(num_balls):
        if max_balls == min_balls: return 2.0
        log_min = np.log(min_balls + 1)
        log_max = np.log(max_balls + 1)
        log_val = np.log(num_balls + 1)
        normalized = (log_val - log_min) / (log_max - log_min) if log_max > log_min else 0
        return 0.5 + normalized * 8.0

    dot = graphviz.Digraph(comment=f'Family Tree of {start_node}')
    dot.attr(rankdir='TB', splines='spline', overlap='false')

    for node in family_graph.nodes():
        node_props = full_graph.nodes[node]
        fixed_color = '#d3e8ff'
        label_text = f"S{int(node[0])} F{int(node[1])}\nBalls: {node_props.get('num_balls', 0)}"
        dot.node(name=str(node), label=label_text, style='filled', fillcolor=fixed_color, shape='box', fontcolor='black', fontsize='10')

    for edge in family_graph.edges():
        source_node, dest_node = edge
        source_props = full_graph.nodes[source_node]
        thickness = map_balls_to_thickness(source_props.get('num_balls', 1))
        dot.edge(str(source_node), str(dest_node), penwidth=str(thickness))
        
    try:
        output_format = os.path.splitext(output_filename)[1][1:] or 'png'
        dot.render(os.path.splitext(output_filename)[0], format=output_format, view=False, cleanup=True)
        return True, f"Success! Image saved to '{output_filename}'"
    except graphviz.backend.ExecutableNotFound:
        return False, "ERROR: Graphviz executable not found. Please install it and add to PATH."
    except Exception as e:
        return False, f"An error occurred during rendering: {e}"

# ==============================================================================
#  GUI APPLICATION
# ==============================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("交互式块体分析器 (v11 - 路径修复)")
        self.geometry("1100x800")
        self.fragments_df = None
        self.genealogy_graph = None
        self.current_step_balls_df = None
        self.current_step_frags_df = None
        self.experiment_path = None
        self.selected_point_artist = None
        
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        main_frame = ttk.Frame(self)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(top_frame, text="实验:").pack(side=tk.LEFT, padx=(5,0))
        self.experiment_var = tk.StringVar()
        experiments = [os.path.basename(d) for d in glob.glob('experiments/*') if os.path.isdir(d)]
        self.exp_dropdown = ttk.Combobox(top_frame, textvariable=self.experiment_var, values=experiments, state="readonly", width=40)
        self.exp_dropdown.pack(side=tk.LEFT, padx=5)
        self.exp_dropdown.bind("<<ComboboxSelected>>", self.load_experiment)
        
        self.analysis_button = ttk.Button(top_frame, text="全局动力学分析", command=self.run_global_analysis, state="disabled")
        self.analysis_button.pack(side=tk.LEFT, padx=(20, 5))

        ttk.Label(top_frame, text="Step:").pack(side=tk.LEFT, padx=(10,0))
        self.step_var = tk.StringVar()
        self.step_dropdown = ttk.Combobox(top_frame, textvariable=self.step_var, state="disabled")
        self.step_dropdown.pack(side=tk.LEFT, padx=5)
        self.step_dropdown.bind("<<ComboboxSelected>>", self.on_step_select)
        
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def load_experiment(self, event=None):
        exp_name = self.experiment_var.get()
        self.experiment_path = os.path.join("experiments", exp_name)
        messagebox.showinfo("正在加载", f"正在为 {exp_name} 加载数据...", parent=self)
        self.fragments_df = load_fragment_data(self.experiment_path)
        if self.fragments_df is None:
            messagebox.showerror("错误", "无法加载块体属性数据 (fragments_properties_step_*.csv)。无法构建谱系。", parent=self)
            return
        self.genealogy_graph = build_genealogy_graph(self.fragments_df)
        steps = sorted(self.fragments_df['step'].unique())
        self.step_dropdown['values'] = steps
        self.step_dropdown.config(state="readonly")
        self.analysis_button.config(state="normal")
        messagebox.showinfo("成功", f"加载了 {len(self.fragments_df)} 条块体记录并构建了谱系图。", parent=self)

    def on_step_select(self, event=None):
        step = int(self.step_var.get())
        self.current_step_balls_df = load_ball_data_for_step(self.experiment_path, step)
        self.current_step_frags_df = self.fragments_df[self.fragments_df['step'] == step]
        if self.current_step_balls_df is None:
            messagebox.showwarning("数据缺失", f"无法为 step {step} 找到或加载球体数据 (fragments_step_{step}.csv)", parent=self)
            self.ax.clear()
            self.canvas.draw()
            return
        self.plot_scene()

    def plot_scene(self):
        self.ax.clear()
        if hasattr(self, 'selected_point_artist') and self.selected_point_artist: self.selected_point_artist = None
        ball_df = self.current_step_balls_df
        frag_df = self.current_step_frags_df
        unique_frags = ball_df['fragment_id'].unique()
        colors = plt.get_cmap('tab20', len(unique_frags))
        frag_color_map = {frag_id: colors(i) for i, frag_id in enumerate(unique_frags)}
        for frag_id, group in ball_df.groupby('fragment_id'):
            color = frag_color_map.get(frag_id, 'gray')
            point_sizes = (group['radius'] * 72 / self.fig.dpi * 2.5)**2
            self.ax.scatter(group['x'], group['y'], s=point_sizes, color=color, alpha=0.8)
        if not frag_df.empty:
            self.ax.scatter(frag_df['centroid_x'], frag_df['centroid_y'], s=40, c='yellow', edgecolors='black', zorder=10, label='质心')
        self.ax.set_title(f"模型视图 Step {self.step_var.get()} (点击质心选择)")
        self.ax.set_xlabel("X 坐标"); self.ax.set_ylabel("Y 坐标")
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.canvas.draw()

    def on_canvas_click(self, event):
        if event.inaxes != self.ax or self.current_step_frags_df is None or self.current_step_frags_df.empty: return
        click_x, click_y = event.xdata, event.ydata
        df = self.current_step_frags_df
        distances = np.sqrt((df['centroid_x'] - click_x)**2 + (df['centroid_y'] - click_y)**2)
        closest_idx = distances.idxmin()
        selected_frag = df.loc[closest_idx]
        frag_id = int(selected_frag['fragment_id'])
        step = int(selected_frag['step'])
        if hasattr(self, 'selected_point_artist') and self.selected_point_artist:
            self.selected_point_artist.remove()
        self.selected_point_artist, = self.ax.plot(selected_frag['centroid_x'], selected_frag['centroid_y'], 'r*', markersize=15, zorder=20)
        self.canvas.draw()
        
        self.prompt_for_plot_type(step, frag_id)
        
    def prompt_for_plot_type(self, step, frag_id):
        win = tk.Toplevel(self)
        win.title("选择谱系图类型")
        win.geometry("350x150")
        
        label = ttk.Label(win, text=f"您已选择块体 S{step} F{frag_id}。\n请选择要生成的谱系图类型：", justify=tk.CENTER)
        label.pack(pady=10)
        
        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=10)

        def on_select(plot_type):
            win.destroy()
            self.generate_family_tree(step, frag_id, plot_type)

        btn_shape = ttk.Button(btn_frame, text="按形状 (固定颜色)", command=lambda: on_select("shape"))
        btn_shape.pack(side=tk.LEFT, padx=10)
        
        btn_thickness = ttk.Button(btn_frame, text="按枝条粗细 (固定颜色)", command=lambda: on_select("thickness"))
        btn_thickness.pack(side=tk.LEFT, padx=10)
        
        win.transient(self)
        win.grab_set()
        self.wait_window(win)
        
    def generate_family_tree(self, step, frag_id, plot_type):
        start_node = (step, frag_id)
        if start_node not in self.genealogy_graph:
            messagebox.showerror("错误", f"块体 {frag_id} 在 step {step} 未在谱系图中找到。", parent=self)
            return

        progress_win = tk.Toplevel(self)
        progress_win.title("正在处理...")
        ttk.Label(progress_win, text=f"正在生成谱系图...\n这可能需要一些时间。").pack(padx=20, pady=20)
        self.update_idletasks()
        
        success, message = False, "未知的绘图类型"
        if plot_type == "shape":
            output_filename = f"family_tree_SHAPES_{self.experiment_var.get()}_S{step}_F{frag_id}.png"
            success, message = plot_family_tree_with_shapes(self.genealogy_graph, start_node, output_filename, self.experiment_path)
        elif plot_type == "thickness":
            output_filename = f"family_tree_THICKNESS_{self.experiment_var.get()}_S{step}_F{frag_id}.png"
            success, message = plot_family_tree_with_variable_thickness(self.genealogy_graph, start_node, output_filename)

        progress_win.destroy()

        if success:
            if messagebox.askyesno("成功", f"{message}\n\n是否要打开图片？", parent=self):
                try: os.startfile(os.path.abspath(output_filename))
                except AttributeError: messagebox.showinfo("信息", "此操作系统不支持自动打开文件。", parent=self)
                except Exception as e: messagebox.showerror("错误", f"无法打开文件: {e}", parent=self)
        else:
            messagebox.showerror("绘图错误", message, parent=self)
            
    def run_global_analysis(self):
        if not self.experiment_path:
            messagebox.showwarning("无实验", "请先选择一个实验。", parent=self)
            return
            
        analyzer_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamics_analyzer_font_final.py")
        
        if not os.path.exists(analyzer_script_path):
            messagebox.showerror("错误", f"分析脚本 'dynamics_analyzer_font_final.py' 未找到。", parent=self)
            return

        python_executable = sys.executable

        try:
            messagebox.showinfo("分析开始", 
                                f"已开始对实验 '{self.experiment_var.get()}' 进行全局动力学分析。\n\n"
                                "这是一个后台进程，可能需要几分钟时间。\n"
                                "完成后，结果将保存在该实验的 'analysis_results' 文件夹中。",
                                parent=self)
            
            subprocess.Popen([python_executable, analyzer_script_path, self.experiment_path])
            
        except Exception as e:
            messagebox.showerror("运行错误", f"启动分析脚本时出错: {e}", parent=self)

if __name__ == "__main__":
    app = App()
    app.mainloop()
