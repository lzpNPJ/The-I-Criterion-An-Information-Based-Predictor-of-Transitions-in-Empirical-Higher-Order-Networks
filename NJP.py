"""
I-Criterion Validation: Publication-Ready Figures
=================================================
Based on original CSV hysteresis data and JSON structural parameters.
Generates:

- Figure 1: I-value distribution (boxplot) + ROC curve + Confusion matrix
- Figure 2: Network-level accuracy bar chart (with I_c overlay) + per-dynamics heatmap
- Figure 3: 7 networks scatter plots (I-value vs control parameter)
- Table 1: Network statistics (N, ⟨k⟩, T, λ_max, Accuracy)

All figures conform to top-journal standards (Arial font, clean layout,
subfigure labels at top edge, no intrusive internal text).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import json
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
import warnings
warnings.filterwarnings('ignore')

# ==================== 顶刊风格设置 ====================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'axes.titlepad': 10,
})

# ==================== 配置 ====================
DATA_PATH = r'C:\Users\DELL\Desktop\I准则论文基础数据+代码\all_networks_hysteresis_30points_20260326_134732'
JSON_PATH = r'C:\Users\DELL\Desktop\I准则论文基础数据+代码\all_networks_paper_20260225_052226\all_results_paper.json'

# 网络列表（与文件夹一致）
networks = [
    'C.elegans_pharynx',
    'Mixed.species_brain_1',
    'Mouse_visual_cortex_1',
    'Mouse_visual_cortex_2',
    'P.pacificus_synaptic_1',
    'Rhesus_brain_2',
    'Rhesus_cerebral_cortex_1'
]

# 完整网络标签
NETWORK_LABELS = {
    'C.elegans_pharynx': 'C. elegans pharynx',
    'Mixed.species_brain_1': 'Mixed species brain',
    'Mouse_visual_cortex_1': 'Mouse visual cortex 1',
    'Mouse_visual_cortex_2': 'Mouse visual cortex 2',
    'P.pacificus_synaptic_1': 'P. pacificus synaptic',
    'Rhesus_brain_2': 'Rhesus brain 2',
    'Rhesus_cerebral_cortex_1': 'Rhesus cerebral cortex 1'
}

dynamics_list = ['sis', 'kuramoto', 'game']
dynamics_names = {'sis': 'SIS', 'kuramoto': 'Kuramoto', 'game': 'Game'}

HYSTERESIS_THRESHOLD = 0.05

# 专业配色
COLORS = {
    'SIS': '#4C72B0',
    'Kuramoto': '#55A868',
    'Game': '#C44E52',
    'line': '#2C3E50',
}

# ==================== I值计算函数 ====================
def compute_I_from_csv(df):
    """从迟滞环CSV计算I值 (I = log(v_info / H))"""
    I_values = []
    r_values = df.iloc[:, 0].values
    low = df['low_init'].values
    high = df['high_init'].values

    for i in range(len(df)):
        p = low[i]

        if p < 1e-10 or p > 0.999:
            H = 0.01
        else:
            H = -p * np.log(p) - (1-p) * np.log(1-p)
            H = max(H, 0.01)

        diff = abs(high[i] - low[i])

        if i > 0:
            dr = abs(r_values[i] - r_values[i-1])
            v_info = diff / dr if dr > 1e-10 else diff
        else:
            v_info = diff

        I = np.log(v_info / H + 1e-10)
        I_values.append(I)

    return I_values


# ==================== 加载结构参数（从JSON）====================
print("="*70)
print("Loading structural parameters from JSON...")
print("="*70)

structural_params = {}
if os.path.exists(JSON_PATH):
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    for item in json_data:
        dataset = item.get('dataset')
        if dataset in networks:
            structural_params[dataset] = {
                'N': item.get('N', '-'),
                '⟨k⟩': item.get('avg_degree', 0),
                'T': item.get('T', 0),
                'λ_max': item.get('lambda_max', 0)
            }
    print(f"  Loaded {len(structural_params)} networks.")
else:
    print(f"  Warning: JSON file not found. Using empty parameters.")

# ==================== 加载CSV数据 ====================
print("\n" + "="*70)
print("Loading hysteresis CSV data...")
print("="*70)

all_samples = []

for network in networks:
    network_path = os.path.join(DATA_PATH, network)
    if not os.path.exists(network_path):
        continue

    for dynamics in dynamics_list:
        csv_path = os.path.join(network_path, f'hysteresis_{dynamics}.csv')
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        I_vals = compute_I_from_csv(df)
        param_vals = df.iloc[:, 0].values
        low = df['low_init'].values
        high = df['high_init'].values
        diff = high - low
        hysteresis_max = np.max(np.abs(diff))
        true_type = 'Explosive' if hysteresis_max > HYSTERESIS_THRESHOLD else 'Continuous'

        for i in range(len(df)):
            all_samples.append({
                'network': network,
                'network_label': NETWORK_LABELS[network],
                'dynamics': dynamics_names[dynamics],
                'param_value': param_vals[i],
                'I_value': I_vals[i],
                'true_type': true_type
            })

df_samples = pd.DataFrame(all_samples)
print(f"  Total samples loaded: {len(df_samples)}")

# ==================== 留一网络交叉验证 ====================
print("\n" + "="*70)
print("Leave-one-network-out cross-validation...")
print("="*70)

groups = df_samples['network'].values
I_array = df_samples['I_value'].values
y_true = (df_samples['true_type'] == 'Explosive').astype(int)

logo = LeaveOneGroupOut()
test_accuracies = []
results_by_network = {}
all_test_pred = []
all_test_true = []

for train_idx, test_idx in logo.split(I_array, y_true, groups=groups):
    test_network = groups[test_idx[0]]

    train_I = I_array[train_idx]
    train_y = y_true[train_idx]
    test_I = I_array[test_idx]
    test_y = y_true[test_idx]

    thresholds = np.linspace(min(train_I)-0.5, max(train_I)+0.5, 500)
    best_thresh = thresholds[0]
    best_acc = 0

    for thresh in thresholds:
        pred = (train_I > thresh).astype(int)
        acc = np.mean(pred == train_y)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    test_pred = (test_I > best_thresh).astype(int)
    test_acc = np.mean(test_pred == test_y)
    test_accuracies.append(test_acc)

    results_by_network[test_network] = {
        'I_c': best_thresh,
        'train_acc': best_acc,
        'test_acc': test_acc,
        'n_correct': int(sum(test_pred == test_y)),
        'n_total': len(test_idx)
    }

    all_test_pred.extend(test_pred)
    all_test_true.extend(test_y)

all_test_pred = np.array(all_test_pred)
all_test_true = np.array(all_test_true)

# ==================== 全局指标 ====================
mean_accuracy = np.mean(test_accuracies)
std_accuracy = np.std(test_accuracies)
cm = confusion_matrix(all_test_true, all_test_pred)
fpr, tpr, _ = roc_curve(all_test_true, I_array)
roc_auc = auc(fpr, tpr)

# ==================== I值统计 ====================
sis_I = df_samples[df_samples['dynamics']=='SIS']['I_value'].values
kura_I = df_samples[df_samples['dynamics']=='Kuramoto']['I_value'].values
game_I = df_samples[df_samples['dynamics']=='Game']['I_value'].values

_, p_sis_kura = mannwhitneyu(sis_I, kura_I)
_, p_sis_game = mannwhitneyu(sis_I, game_I)
_, p_kura_game = mannwhitneyu(kura_I, game_I)

# ==================== 构建准确率矩阵（7×3） ====================
net_list = networks
dyn_list = ['SIS', 'Kuramoto', 'Game']
acc_matrix = np.zeros((len(net_list), len(dyn_list)))

for i, net in enumerate(net_list):
    I_c = results_by_network[net]['I_c']
    net_samples = df_samples[df_samples['network'] == net]
    for j, dyn in enumerate(dyn_list):
        dyn_samples = net_samples[net_samples['dynamics'] == dyn]
        if len(dyn_samples) > 0:
            pred = (dyn_samples['I_value'].values > I_c).astype(int)
            true = (dyn_samples['true_type'] == 'Explosive').astype(int)
            acc_matrix[i, j] = np.mean(pred == true)

print(f"\nMean test accuracy: {mean_accuracy:.1%}")

# ==================== Figure 1: 箱线图 + ROC + 混淆矩阵 ====================
print("\nGenerating Figure 1...")
fig1 = plt.figure(figsize=(14, 4.5))
gs1 = gridspec.GridSpec(1, 3, figure=fig1, wspace=0.35)

# ---- 1a: I值分布箱线图 ----
ax1a = fig1.add_subplot(gs1[0])
data_by_dyn = [sis_I, kura_I, game_I]
positions = [1, 2, 3]
colors_box = [COLORS['SIS'], COLORS['Kuramoto'], COLORS['Game']]

bp = ax1a.boxplot(data_by_dyn, positions=positions, widths=0.55,
                   patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 手动添加散点
for i, data in enumerate(data_by_dyn):
    x = np.random.normal(positions[i], 0.04, size=min(len(data), 150))
    ax1a.scatter(x, np.random.choice(data, min(len(data), 150)),
                s=6, alpha=0.15, color=colors_box[i])

# 显著性标记
def add_sig(ax, x1, x2, y, p):
    if p < 0.001:
        symbol = '***'
    elif p < 0.01:
        symbol = '**'
    elif p < 0.05:
        symbol = '*'
    else:
        return
    ax.plot([x1, x1, x2, x2], [y, y+0.5, y+0.5, y], 'k-', lw=0.8)
    ax.text((x1+x2)/2, y+0.8, symbol, ha='center', fontsize=10)

y_max = max(np.max(sis_I), np.max(kura_I), np.max(game_I))
add_sig(ax1a, 1, 2, y_max + 0.5, p_sis_kura)
add_sig(ax1a, 1, 3, y_max + 1.2, p_sis_game)
add_sig(ax1a, 2, 3, y_max + 0.5, p_kura_game)

ax1a.set_xticks(positions)
ax1a.set_xticklabels(['SIS', 'Kuramoto', 'Game'])
ax1a.set_ylabel('I-criterion value')
ax1a.set_title('a', loc='left', fontsize=12, fontweight='bold', pad=10)
ax1a.grid(True, alpha=0.2, axis='y')
ax1a.set_ylim(-25, 18)

# ---- 1b: ROC曲线 ----
ax1b = fig1.add_subplot(gs1[1])
ax1b.plot(fpr, tpr, color=COLORS['line'], lw=2.5, label=f'AUC = {roc_auc:.3f}')
ax1b.plot([0, 1], [0, 1], color='#95A5A6', lw=1.5, linestyle='--')
ax1b.set_xlabel('False Positive Rate')
ax1b.set_ylabel('True Positive Rate')
ax1b.set_title('b', loc='left', fontsize=12, fontweight='bold', pad=10)
ax1b.legend(loc='lower right', frameon=True)
ax1b.grid(True, alpha=0.2)

# ---- 1c: 混淆矩阵 ----
ax1c = fig1.add_subplot(gs1[2])
im = ax1c.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
ax1c.set_xticks([0, 1])
ax1c.set_yticks([0, 1])
ax1c.set_xticklabels(['Continuous', 'Explosive'])
ax1c.set_yticklabels(['Continuous', 'Explosive'])
ax1c.set_xlabel('Predicted')
ax1c.set_ylabel('True')
ax1c.set_title('c', loc='left', fontsize=12, fontweight='bold', pad=10)
total = np.sum(cm)
for i in range(2):
    for j in range(2):
        pct = cm[i, j] / total * 100
        ax1c.text(j, i, f'{cm[i, j]}\n({pct:.1f}%)',
                 ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=ax1c, shrink=0.8)

plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'Figure1.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(DATA_PATH, 'Figure1.png'), dpi=300, bbox_inches='tight')
print("  ✅ Figure 1 saved")

# ==================== Figure 2: 网络准确率 + 热力图 ====================
print("\nGenerating Figure 2...")
fig2 = plt.figure(figsize=(12, 4.5))
gs2 = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.6)  # 增大间距

# ---- 2a: 网络准确率条形图（含I_c叠加） ----
ax2a = fig2.add_subplot(gs2[0])
net_names = [NETWORK_LABELS[n] for n in net_list]
x_pos = np.arange(len(net_names))
test_accs = [results_by_network[n]['test_acc'] for n in net_list]
I_c_vals = [results_by_network[n]['I_c'] for n in net_list]

# 柱状图：准确率
bars = ax2a.bar(x_pos, test_accs, color=COLORS['Game'], alpha=0.7,
                edgecolor='black', width=0.6, label='Test Accuracy')

# 折线：I_c（使用右侧Y轴）
ax2a_twin = ax2a.twinx()
ax2a_twin.plot(x_pos, I_c_vals, 'o-', color=COLORS['SIS'], linewidth=2,
               markersize=6, markerfacecolor='white', markeredgewidth=1.5,
               label='Optimal I$_c$')

# 设置轴标签
ax2a.set_xticks(x_pos)
ax2a.set_xticklabels(net_names, rotation=45, ha='right', fontsize=8)
ax2a.set_ylabel('Test Accuracy', fontsize=10)
ax2a.set_ylim(0, 1)
ax2a_twin.set_ylabel('Optimal I$_c$', fontsize=10)

ax2a.set_title('a', loc='left', fontsize=12, fontweight='bold', pad=10)

# 平均准确率线
mean_acc = np.mean(test_accs)
ax2a.axhline(y=mean_acc, color='black', linestyle='--', linewidth=1.5,
             label=f'Mean = {mean_acc:.1%}')

# 图例：分开放置避免重叠
ax2a.legend(loc='upper left', frameon=True, fontsize=8)
ax2a_twin.legend(loc='lower right', frameon=True, fontsize=8)

ax2a.grid(True, alpha=0.2, axis='y')

# ---- 2b: 热力图 ----
ax2b = fig2.add_subplot(gs2[1])
im2 = ax2b.imshow(acc_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

ax2b.set_xticks([0, 1, 2])
ax2b.set_yticks(range(len(net_names)))
ax2b.set_xticklabels(['SIS', 'Kuramoto', 'Game'])
ax2b.set_yticklabels(net_names, fontsize=8)
ax2b.set_xlabel('Dynamics', fontsize=10)
ax2b.set_ylabel('Network', fontsize=10)
ax2b.set_title('b', loc='left', fontsize=12, fontweight='bold', pad=10)

for i in range(len(net_names)):
    for j in range(3):
        ax2b.text(j, i, f'{acc_matrix[i, j]:.0%}',
                 ha='center', va='center', color='black', fontsize=8)
plt.colorbar(im2, ax=ax2b, shrink=0.7)

plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'Figure2.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(DATA_PATH, 'Figure2.png'), dpi=300, bbox_inches='tight')
print("  ✅ Figure 2 saved")

# ==================== Figure 3: 所有7个网络散点图 ====================
print("\nGenerating Figure 3...")
fig3 = plt.figure(figsize=(14, 12))
gs3 = gridspec.GridSpec(3, 3, figure=fig3, hspace=0.35, wspace=0.35)

colors_scatter = {'SIS': COLORS['SIS'], 'Kuramoto': COLORS['Kuramoto'], 'Game': COLORS['Game']}
markers = {'SIS': 'o', 'Kuramoto': 's', 'Game': '^'}

for idx, network in enumerate(net_list):
    row = idx // 3
    col = idx % 3
    ax = fig3.add_subplot(gs3[row, col])

    net_data = df_samples[df_samples['network'] == network]

    for dyn in ['SIS', 'Kuramoto', 'Game']:
        dyn_data = net_data[net_data['dynamics'] == dyn]
        if len(dyn_data) > 0:
            ax.scatter(dyn_data['param_value'], dyn_data['I_value'],
                      c=colors_scatter[dyn], marker=markers[dyn], s=12,
                      alpha=0.4, edgecolors='none', label=dyn if idx == 0 else "")

    I_c = results_by_network[network]['I_c']
    ax.axhline(y=I_c, color='black', linestyle='--', linewidth=1, alpha=0.7)

    acc = results_by_network[network]['test_acc']
    ax.text(0.05, 0.92, f'accuracy: {acc:.0%}', transform=ax.transAxes,
            fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_title(NETWORK_LABELS[network], fontsize=9)
    ax.set_xlabel('Control parameter', fontsize=8)
    ax.set_ylabel('I-value', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, linestyle='--')

# 全局图例
handles = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor=c,
                      markersize=8, label=d)
           for d, c, m in zip(['SIS', 'Kuramoto', 'Game'],
                              [colors_scatter['SIS'], colors_scatter['Kuramoto'], colors_scatter['Game']],
                              [markers['SIS'], markers['Kuramoto'], markers['Game']])]
handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='I$_c$'))
fig3.legend(handles=handles, loc='upper center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.97))

fig3.suptitle('I-value vs Control Parameter Across Networks', fontsize=12, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(DATA_PATH, 'Figure3.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(DATA_PATH, 'Figure3.png'), dpi=300, bbox_inches='tight')
print("  ✅ Figure 3 saved")

# ==================== Table 1: 网络统计表 ====================
print("\nGenerating Table 1...")

table_data = []
for net in net_list:
    params = structural_params.get(net, {})
    table_data.append({
        'Network': NETWORK_LABELS[net],
        'N': params.get('N', '-'),
        '⟨k⟩': f"{params.get('⟨k⟩', 0):.2f}",
        'T': f"{params.get('T', 0):.4f}",
        'λ_max': f"{params.get('λ_max', 0):.2f}",
        'Accuracy': f"{results_by_network[net]['test_acc']:.1%}"
    })

df_table = pd.DataFrame(table_data)

latex = r"\begin{table}[ht]" + "\n"
latex += r"\centering" + "\n"
latex += r"\caption{Network statistics and I-criterion test accuracy}" + "\n"
latex += r"\begin{tabular}{lcccccc}" + "\n"
latex += r"\hline" + "\n"
latex += r"Network & $N$ & $\langle k\rangle$ & $T$ & $\lambda_{\max}$ & Accuracy \\" + "\n"
latex += r"\hline" + "\n"
for d in table_data:
    latex += f"{d['Network']} & {d['N']} & {d['⟨k⟩']} & {d['T']} & {d['λ_max']} & {d['Accuracy']} \\\\\n"
latex += r"\hline" + "\n"
latex += r"\end{tabular}" + "\n"
latex += r"\label{tab:network_stats}" + "\n"
latex += r"\end{table}" + "\n"

with open(os.path.join(DATA_PATH, 'Table1.tex'), 'w', encoding='utf-8') as f:
    f.write(latex)
df_table.to_csv(os.path.join(DATA_PATH, 'Table1.csv'), index=False)
print("  ✅ Table 1 saved")

# ==================== 最终总结 ====================
print("\n" + "="*70)
print("✅ All figures and tables generated successfully!")
print("="*70)
print(f"Total samples: {len(df_samples)}")
print(f"Mean test accuracy: {mean_accuracy:.1%}")
print(f"Output directory: {DATA_PATH}")
print("\nFiles saved:")
print("  - Figure1.pdf/png")
print("  - Figure2.pdf/png")
print("  - Figure3.pdf/png")
print("  - Table1.tex/csv")
print("="*70)

plt.show()