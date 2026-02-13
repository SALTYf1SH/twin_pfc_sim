# -*- coding: utf-8 -*-
"""
Minimal Test Script for Stress Utility Function (Version 3)

This script tests the 'get_avg_ball_yy_stress' function from 'utils.py'.
It assumes a PFC model is already loaded.

Key changes:
- Corrected stress access from '.yy' to array index '[1, 1]'.
"""

import itasca
from itasca import ball
import sys
import os
import traceback
import importlib  # 导入 importlib 库

# 确保 'utils.py' 可以被找到
sys.path.append(os.getcwd())

# --- 1. 导入并强制重新加载 'utils' 模块 ---
try:
    # 正常导入
    import utils
    # 强制PFC重新加载磁盘上的最新版本
    importlib.reload(utils) 
    print("INFO: Forcing reload of 'utils.py' module.")
    
    # 现在才从重新加载后的模块中导入新函数
    from utils import get_avg_ball_yy_stress
    print("INFO: Successfully imported 'get_avg_ball_yy_stress' from reloaded utils.py.")

except ImportError:
    print("="*50)
    print("FATAL ERROR: Could not import 'get_avg_ball_yy_stress' EVEN after reload.")
    print(f"Current working directory: {os.getcwd()}")
    print("Please double-check that 'utils.py' is saved and contains the new function.")
    print("="*50)
    raise SystemExit() 
except Exception as e:
    print(f"FATAL ERROR during import: {e}")
    traceback.print_exc()
    raise SystemExit() 


print("--- Stress Utility Test Script ---")

# --- 2. 获取模型中的所有球 ---
all_balls = list(ball.list())

if not all_balls:
    print("="*50)
    print("ERROR: No balls found in the current model.")
    print("Please load a model (e.g., 'model restore jiaojie.sav') BEFORE running this script.")
    print("="*50)
    raise SystemExit() 

print(f"INFO: Found {len(all_balls)} balls in the model.")

# --- 3. 选中第一个球 ---
first_ball = all_balls[0]
print(f"INFO: Selecting the first ball (ID: {first_ball.id()}) for testing.")
print(f"       Ball Position: (x={first_ball.pos_x():.2f}, y={first_ball.pos_y():.2f})")

# --- 4. 准备函数输入 ---
# 函数 get_avg_ball_yy_stress 期望一个列表(list)作为输入
test_ball_list = [first_ball]

# --- 5. 执行测试 ---
print("INFO: Calling get_avg_ball_yy_stress()...")
try:
    # 调用函数
    avg_yy_stress = get_avg_ball_yy_stress(test_ball_list)
    
    # 6. 打印结果
    print("\n" + "="*50)
    print("--- TEST RESULT ---")
    print(f"Function executed successfully.")
    print(f"Average YY-Stress for ball ID {first_ball.id()}: {avg_yy_stress}")

    # 额外验证：直接获取该球的应力值进行对比
    # --- 这是修改后的行 ---
    direct_stress = first_ball.stress()[1, 1]
    # --- 修改结束 ---
    print(f"(Direct value from ball.stress()[1, 1]: {direct_stress})")
    
    if avg_yy_stress == direct_stress:
        print("\nSUCCESS: Function result matches direct value.")
    else:
        print("\nWARNING: Function result does NOT match direct value. Check logic.")

except Exception as e:
    print("\n" + "="*50)
    print(f"--- TEST FAILED ---")
    print(f"An error occurred while calling the function: {e}")
    traceback.print_exc()

print("="*50)
print("--- Test Complete ---")