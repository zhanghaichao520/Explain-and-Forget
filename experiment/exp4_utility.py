import matplotlib.pyplot as plt
import numpy as np

# 随机生成数据 (每个子图随机生成)
np.random.seed(0)
data = {
    'HR@K on LightGCN': {
        'Origin Model': [0.331,0.5333,0.6363 ],
        'CEU-Full': [0.3308, 0.531, 0.6353],
        'CEU-Random': [0.2326 , 0.4526,0.5748 ],
        'CEU-Reverse': [0.2351, 0.4684,  0.5914],

    },
    'NDCG@K on LightGCN': {
        'Origin Model': [0.331,0.2932, 0.279 ],
        'CEU-Full': [0.3308 , 0.2918 , 0.2781],
        'CEU-Random': [0.2326 , 0.2152 , 0.2109 ],
        'CEU-Reverse': [0.2351, 0.2315 , 0.2336 ],
    },
    'HR@K on BPR': {
        'Origin Model': [0.3142, 0.5227 , 0.62 ],
        'CEU-Full': [0.3124 , 0.5154 ,0.6149],
        'CEU-Random': [0.2202 ,0.4265 ,  0.5472],
        'CEU-Reverse': [0.2354, 0.4591 ,  0.58 ],
    },
    'NDCG@K on BPR': {
        'Origin Model': [0.3142 , 0.2821 , 0.2657 ],
        'CEU-Full': [0.3124 , 0.279 , 0.2629],
        'CEU-Random': [0.2202 , 0.1999, 0.195],
        'CEU-Reverse': [0.2354, 0.2238 , 0.2244 ],
    },
}

pc_ratios = ['TopK=1', 'TopK=3', 'TopK=5']
datasets = list(data.keys())
methods = ['Origin Model', 'CEU-Full', 'CEU-Random', 'CEU-Reverse']

# 设置字体和样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['legend.fontsize'] = 16

# 调整figsize，增加画布宽度
fig, axes = plt.subplots(1, 4, figsize=(22, 5))  # 4个子图横着排列

# 颜色设置，使用高对比度的淡色
bar_colors = ['#A9A9A9','#0072B2', '#E69F00', '#D55E00', '#D8BFD8']  # Light blue, light pink, light green, light purple
# bar_colors = ['#7F7F7F','#1F77B4', '#FF7F0E', '#2CA02C', '#D8BFD8']  # Light blue, light pink, light green, light purple

# 图例标签
legend_labels_bar = [f'{method}' for method in methods]

# 创建图例对象
bar_handles, bar_labels = [], []

for i, dataset in enumerate(datasets):
    ax = axes[i]  # 获取当前子图

    x = np.arange(len(pc_ratios))
    width = 0.15  # 调整宽度使每组柱子为4个

    # 绘制柱状图
    for j, method in enumerate(methods):
        bars = ax.bar(x + j * width, data[dataset][method], width, color=bar_colors[j], edgecolor='black', label=legend_labels_bar[j] if i == 0 else "")
        if i == 0:  # 只为第一个子图添加图例
            bar_handles.append(bars[0])  # 取第一个条形图句柄
            bar_labels.append(legend_labels_bar[j])

    # 设置 x 轴刻度和标签
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(pc_ratios)

    # 设置 y 轴标签
    if i == 0 or i == 2:
        ax.set_ylabel('HR')
        ax.set_ylim(0.15, 0.7)
    if i == 1 or i == 3:
        ax.set_ylabel('NDCG')
        ax.set_ylim(0.15, 0.35)

    # 设置标题
    ax.set_title(dataset)

    # 设置 y 轴范围 (根据图片目测估算，请根据实际数据调整)
    # if dataset == 'ML-1M (BPR)':
    #     ax.set_ylim(0, 0.3)
    # if dataset == 'ML-1M (LightGCN)':
    #     ax.set_ylim(0, 0.35)
    # 设置网格线，增强可读性
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# 使用 subplots_adjust 精确控制子图间距
plt.subplots_adjust(left=0.07, right=0.92, wspace=0.2, top=0.8)  # 调整左右边界和水平间距

# 在上方居中显示图例，调整 y 值使其远离柱状图, 并添加边框
legend = plt.legend(handles=bar_handles, labels=bar_labels,
           loc='best', bbox_to_anchor=(1, 1.25), ncol=5, frameon=True, edgecolor='gray')

# plt.savefig('exp4_utility.svg', format='svg')
plt.savefig("exp4_utility.pdf", bbox_inches='tight')
# 显示图表
plt.show()
