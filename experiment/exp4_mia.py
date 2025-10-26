import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager

def setup_academic_style():
    """
    为论文图表设置专业、清晰的 Matplotlib 样式。
    """
    try:
        # 尝试使用 Times New Roman 字体
        matplotlib.font_manager.findfont("Times New Roman")
        plt.rcParams['font.family'] = 'Times New Roman'
    except:
        print("未找到 Times New Roman 字体，使用默认 serif 字体。")
        plt.rcParams['font.family'] = 'serif'
        
    plt.rcParams['font.size'] = 16  # 增大基础字体大小
    plt.rcParams['axes.labelsize'] = 18  # X, Y 轴标签大小
    plt.rcParams['axes.titlesize'] = 20  # 子图标题大小
    plt.rcParams['legend.fontsize'] = 16  # 图例大小
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['axes.linewidth'] = 1.5 # 坐标轴线宽
    plt.rcParams['lines.markersize'] = 12 # 标记大小
    plt.rcParams['lines.markeredgewidth'] = 2.5 # 标记边缘宽度
    plt.rcParams['figure.figsize'] = (28, 7) # 调整图表总尺寸以适应1x4布局

def plot_tradeoff_scatter_enhanced(ax, data, title, utility_metric, is_bpr=False, show_ylabel=False):
    """
    绘制单个增强版的权衡散点图。
    utility_metric: 'ndcg@10' 或 'hr@10'
    is_bpr: BPR的MIA ACC值域不同，需要调整Y轴偏移量
    show_ylabel: 是否显示Y轴标签
    """
    # --- 方法、颜色和标记 ---
    methods = ['Origin', 'CEU-Full', 'CEU-Random', 'CEU-Reverse']
    # 颜色 (灰色, 蓝色, 橙色, 红色)
    colors = ['#A9A9A9', '#0072B2', '#E69F00', '#D55E00']
    # 标记 (圆形, 方形, 三角形, 叉形)
    markers = ['o', 's', '^', 'X']
    
    unlearning_metric = 'MIA ACC'

    points = {}
    
    # 1. 绘制散点
    for i, method in enumerate(methods):
        x = data[method][utility_metric]
        y = data[method][unlearning_metric]
        points[method] = (x, y)
        
        ax.scatter(x, y, 
                   color=colors[i], 
                   marker=markers[i], 
                   s=300,  # 增大标记尺寸
                   label=method, 
                   alpha=0.9,
                   edgecolors='black', # 黑色描边
                   linewidth=1.5,
                   zorder=10) # 确保点在辅助线之上

    # 2. 添加文本标签（代替图例）
    if 'ndcg' in utility_metric:
        offsets_x = {'Origin': 0.0005, 'CEU-Full': 0.0005, 'CEU-Random': -0.001, 'CEU-Reverse': -0.001}
    else: # 'hr'
        offsets_x = {'Origin': 0.0008, 'CEU-Full': 0.0008, 'CEU-Random': -0.0012, 'CEU-Reverse': -0.0012}

    if is_bpr:
        offsets_y = {'Origin': 0.01, 'CEU-Full': 0.02, 'CEU-Random': 0.02, 'CEU-Reverse': -0.025}
        # BPR的Random和Full的MIA几乎重合，特殊处理
        if method == 'CEU-Reverse' and utility_metric == 'ndcg@10':
             offsets_y['CEU-Reverse'] = 0.005
        if method == 'CEU-Full' and utility_metric == 'ndcg@10':
             offsets_y['CEU-Full'] = -0.015
    else: # LightGCN
        offsets_y = {'Origin': 0.01, 'CEU-Full': 0.03, 'CEU-Random': 0.02, 'CEU-Reverse': 0.03}

    for method, (x, y) in points.items():
        ax.text(x + offsets_x[method], y + offsets_y[method], method, 
                fontsize=14, 
                fontweight='bold', 
                va='center',
                color=colors[methods.index(method)])

    # --- 3. (关键修改) 添加辅助线（象限） ---
    ceu_full_x, ceu_full_y = points['CEU-Full']
    origin_x, origin_y = points['Origin']
    
    # 垂直线 (区分高效用 vs 低效用)
    # **新逻辑**: 找到 "Random" 和 "Reverse" 中效用最高的那个
    max_utility_of_ablations = max(points['CEU-Random'][0], points['CEU-Reverse'][0])
    # 将垂线画在 CEU-Full 和 (Random/Reverse的最大值) 之间
    mid_x = (ceu_full_x + max_utility_of_ablations) / 2 - 0.003
    ax.axvline(x=mid_x, color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=5)
    
    # 水平线 (区分高MIA vs 低MIA)
    # **新逻辑**: 找到 "Full", "Random", "Reverse" 中MIA最高的那个
    max_mia_of_unlearned = max(ceu_full_y, points['CEU-Random'][1], points['CEU-Reverse'][1])
    # 将水平线画在 Origin 和 (所有遗忘方法的最高MIA) 之间
    mid_y = (origin_y + max_mia_of_unlearned) / 2
    ax.axhline(y=mid_y, color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=5)

    # 4. 添加象限文本注释
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    x_pos_right = mid_x + (xmax - mid_x) * 0.5
    x_pos_left = xmin + (mid_x - xmin) * 0.5
    y_pos_up = mid_y + (ymax - mid_y) * 0.5
    y_pos_down = ymin + (mid_y - ymin) * 0.5
    
    ax.text(x_pos_right+0.002, y_pos_up, 'Not Unlearned\n(High Utility, High MIA)', 
             ha='center', va='center', fontsize=15, color='gray', alpha=0.8)
    ax.text(x_pos_left-0.005, y_pos_up, 'Worst Case\n(Low Utility, High MIA)', 
             ha='center', va='center', fontsize=15, color='gray', alpha=0.8)
    ax.text(x_pos_left-0.005, y_pos_down, 'Destructive Unlearning\n(Low Utility, Low MIA)', 
             ha='center', va='center', fontsize=15, color='red', alpha=0.8)
    ax.text(x_pos_right+0.002, y_pos_down, '**Ideal Region**\n(High Utility, Low MIA)', 
             ha='center', va='center', fontsize=15, fontweight='bold', color='green', alpha=0.8)


    # --- 标题和标签 ---
    ax.set_title(title, weight='bold', fontsize=20)
    ax.set_xlabel(f'Model Utility ({utility_metric.upper()}) $\\rightarrow$ (Better)', fontsize=18)
    
    # --- 修正 ---
    # 根据传入参数决定是否显示Y轴标签
    if show_ylabel:
        if is_bpr:
            ax.set_ylabel(f'MIA ACC (on BPR) ($\\downarrow$ Better)', fontsize=18)
        else:
            ax.set_ylabel(f'MIA ACC (on LightGCN) ($\\downarrow$ Better)', fontsize=18)

    ax.grid(True, linestyle=':', alpha=0.5)

# --- 准备数据 ---
plot_data = {
    'LightGCN': {
        'Origin': {'ndcg@10': 0.2682, 'hr@10': 0.7603, 'MIA ACC': 0.9678},
        'CEU-Full': {'ndcg@10': 0.2671, 'hr@10': 0.7584, 'MIA ACC': 0.4789},
        'CEU-Random': {'ndcg@10': 0.2161, 'hr@10': 0.7295, 'MIA ACC': 0.4755},
        'CEU-Reverse': {'ndcg@10': 0.2412, 'hr@10': 0.7442, 'MIA ACC': 0.4783}
    },
    'BPR': {
        'Origin': {'ndcg@10': 0.2554, 'hr@10': 0.7417, 'MIA ACC': 0.9619},
        'CEU-Full': {'ndcg@10': 0.2533, 'hr@10': 0.7373, 'MIA ACC': 0.6847},
        'CEU-Random': {'ndcg@10': 0.2022, 'hr@10': 0.7073, 'MIA ACC': 0.6842},
        'CEU-Reverse': {'ndcg@10': 0.2310, 'hr@10': 0.7290, 'MIA ACC': 0.6847}
    }
}
# 您可以在这里轻松切换 K 值
K_VALUE_UTILITY = '10' 
metric_hr = f'hr@{K_VALUE_UTILITY}'
metric_ndcg = f'ndcg@{K_VALUE_UTILITY}'

# --- 开始绘图 ---
setup_academic_style()
# 创建 1x4 子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- (a) HR on LightGCN ---
plot_tradeoff_scatter_enhanced(ax1, plot_data['LightGCN'], 
                               f'(a) HR@{K_VALUE_UTILITY} vs. MIA on LightGCN', 
                               utility_metric=metric_hr, is_bpr=False, show_ylabel=True)
ax1.set_xlim(0.72, 0.77) # LightGCN HR 范围
ax1.set_ylim(0.45, 1.0) # LightGCN MIA 范围

# --- (b) HR on BPR ---
plot_tradeoff_scatter_enhanced(ax2, plot_data['BPR'], 
                               f'(b) HR@{K_VALUE_UTILITY} vs. MIA on BPR', 
                               utility_metric=metric_hr, is_bpr=True, show_ylabel=True)
ax2.set_xlim(0.70, 0.75) # BPR HR 范围
ax2.set_ylim(0.65, 1.0) # BPR MIA 范围

# # --- (c) NDCG on LightGCN ---
# plot_tradeoff_scatter_enhanced(ax3, plot_data['LightGCN'], 
#                                f'(c) NDCG@{K_VALUE_UTILITY} vs. MIA on LightGCN', 
#                                utility_metric=metric_ndcg, is_bpr=False, show_ylabel=False)
# ax3.set_xlim(0.21, 0.28) # LightGCN NDCG 范围
# ax3.set_ylim(0.45, 1.0) # 保持Y轴与(a)一致

# # --- (d) NDCG on BPR ---
# plot_tradeoff_scatter_enhanced(ax4, plot_data['BPR'], 
#                                f'(d) NDCG@{K_VALUE_UTILITY} vs. MIA on BPR', 
#                                utility_metric=metric_ndcg, is_bpr=True, show_ylabel=False)
# ax4.set_xlim(0.19, 0.27) # BPR NDCG 范围
# ax4.set_ylim(0.65, 1.0) # 保持Y轴与(b)一致


# --- 布局和保存 ---
plt.tight_layout()

# --- 创建共享图例 ---
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))
plt.subplots_adjust(bottom=0.25) 

output_filename = "exp4_mia.pdf"
plt.savefig(output_filename, bbox_inches='tight')

# plt.show()