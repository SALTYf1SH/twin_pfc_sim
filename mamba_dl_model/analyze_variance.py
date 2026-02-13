import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results
baseline_df = pd.read_csv('robustness_baseline_results.csv')
ablation_df = pd.read_csv('robustness_mamba_ablation_results.csv')

# Focus on stress task
stress_baseline = baseline_df[baseline_df['Task'] == 'stress'].copy()

# Combine all models for comparison
comparison_data = []

for _, row in stress_baseline.iterrows():
    comparison_data.append({
        'Model': row['Model'],
        'Metric': row['Metric'],
        'Mean': row['Mean'],
        'Std': row['Std'],
        'CI_Lower': row['CI_Lower'],
        'CI_Upper': row['CI_Upper'],
        'CI_Width': row['CI_Upper'] - row['CI_Lower'],
        'CV': row['Std'] / abs(row['Mean']) if row['Mean'] != 0 else 0  # Coefficient of Variation
    })

comparison_df = pd.DataFrame(comparison_data)

# Print detailed analysis
print("=" * 80)
print("应力反演任务 - 模型性能方差分析")
print("=" * 80)
print()

metrics_of_interest = ['PCC', 'MAE', 'MSE', 'Evo']

for metric in metrics_of_interest:
    print(f"\n{'='*80}")
    print(f"指标: {metric}")
    print(f"{'='*80}")
    
    metric_data = comparison_df[comparison_df['Metric'] == metric].sort_values('Mean')
    
    print(f"\n{'模型':<15} {'均值':<12} {'标准差':<12} {'变异系数':<12} {'置信区间宽度':<15} {'CI下限':<12} {'CI上限':<12}")
    print("-" * 100)
    
    for _, row in metric_data.iterrows():
        print(f"{row['Model']:<15} {row['Mean']:<12.6f} {row['Std']:<12.6f} {row['CV']:<12.4f} "
              f"{row['CI_Width']:<15.6f} {row['CI_Lower']:<12.6f} {row['CI_Upper']:<12.6f}")
    
    # Analyze variance ranking
    print(f"\n方差排名 (从高到低):")
    variance_rank = metric_data.sort_values('Std', ascending=False)
    for i, (_, row) in enumerate(variance_rank.iterrows(), 1):
        print(f"  {i}. {row['Model']}: Std={row['Std']:.6f}, CV={row['CV']:.4f}")
    
    # Check if Mamba CI upper bound covers other models
    mamba_row = metric_data[metric_data['Model'] == 'MAMBA'].iloc[0]
    print(f"\nMamba置信区间分析:")
    print(f"  Mamba CI: [{mamba_row['CI_Lower']:.6f}, {mamba_row['CI_Upper']:.6f}]")
    
    for _, row in metric_data.iterrows():
        if row['Model'] != 'MAMBA':
            if mamba_row['CI_Upper'] >= row['CI_Upper']:
                print(f"  ⚠️  Mamba CI上限 ({mamba_row['CI_Upper']:.6f}) >= {row['Model']} CI上限 ({row['CI_Upper']:.6f})")
            
            # Check if means are significantly different
            if metric in ['MAE', 'MSE', 'Evo']:  # Lower is better
                if mamba_row['Mean'] > row['Mean']:
                    print(f"  📊 Mamba均值 ({mamba_row['Mean']:.6f}) 劣于 {row['Model']} ({row['Mean']:.6f})")
            else:  # Higher is better (PCC)
                if mamba_row['Mean'] < row['Mean']:
                    print(f"  📊 Mamba均值 ({mamba_row['Mean']:.6f}) 劣于 {row['Model']} ({row['Mean']:.6f})")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('应力反演任务 - 模型性能方差对比', fontsize=16, fontweight='bold')

for idx, metric in enumerate(metrics_of_interest):
    ax = axes[idx // 2, idx % 2]
    
    metric_data = comparison_df[comparison_df['Metric'] == metric].sort_values('Mean')
    
    models = metric_data['Model'].values
    means = metric_data['Mean'].values
    ci_lower = metric_data['CI_Lower'].values
    ci_upper = metric_data['CI_Upper'].values
    
    # Plot error bars
    x_pos = np.arange(len(models))
    colors = ['#1f77b4' if m != 'MAMBA' else '#d62728' for m in models]
    
    ax.errorbar(x_pos, means, 
                yerr=[means - ci_lower, ci_upper - means],
                fmt='o', markersize=10, capsize=8, capthick=2,
                ecolor=colors, markerfacecolor=colors, markeredgecolor='black',
                linewidth=2, alpha=0.8)
    
    # Highlight Mamba
    mamba_idx = list(models).index('MAMBA')
    ax.axhline(y=ci_upper[mamba_idx], color='red', linestyle='--', alpha=0.5, 
               label=f'Mamba CI Upper: {ci_upper[mamba_idx]:.6f}')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} - Mean ± 95% CI', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Add value labels
    for i, (m, mean, std) in enumerate(zip(models, means, metric_data['Std'].values)):
        ax.text(i, mean, f'{mean:.4f}\n(σ={std:.4f})', 
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('variance_analysis_stress.png', dpi=300, bbox_inches='tight')
print(f"\n\n可视化图表已保存至: variance_analysis_stress.png")

# Summary statistics
print(f"\n\n{'='*80}")
print("总结统计")
print(f"{'='*80}")

print("\n各模型平均变异系数 (CV) - 越小越稳定:")
cv_summary = comparison_df.groupby('Model')['CV'].mean().sort_values()
for model, cv in cv_summary.items():
    print(f"  {model}: {cv:.4f}")

print("\n各模型平均置信区间宽度 - 越小越稳定:")
ci_summary = comparison_df.groupby('Model')['CI_Width'].mean().sort_values()
for model, ci_width in ci_summary.items():
    print(f"  {model}: {ci_width:.6f}")
