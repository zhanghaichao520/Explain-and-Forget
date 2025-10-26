import matplotlib.pyplot as plt
import numpy as np

# 原始数据
alpha_original = np.array([0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 9])
# 用于绘图的均匀分布横轴
alpha = np.linspace(0.1, 9, len(alpha_original))
ml1m_LGN_mia_acc = np.array([0.4991, 0.4992, 0.4994, 0.4996, 0.4999, 0.5008, 0.5022, 0.5025, 0.5013])
ml1m_LGN_hr_20 = np.array([0.7599,0.7573,0.7553,0.75,0.7422,0.7071,0.7022,0.7022,0.702])
ml1m_LGN_ndcg_20 = np.array([0.2678,0.2664,0.2637,0.2598,0.2543,0.2333,0.2305,0.2302,0.23])

ml1m_BPR_mia_acc = np.array([0.5083, 0.5083, 0.5083, 0.5085, 0.5082, 0.5084, 0.5085, 0.5085, 0.5085])
ml1m_BPR_hr_20 = np.array([0.7396,0.7402,0.7364,0.7298,0.7189,0.6863,0.6829,0.6833,0.6833])
ml1m_BPR_ndcg_20 = np.array([0.2549,0.2532,0.2497,0.2456,0.2392,0.2198,0.2183,0.2179,0.2178])

netflix_LGN_mia_acc = np.array([0.4991, 0.4992, 0.4994, 0.4996, 0.4999, 0.5008, 0.5022, 0.5025, 0.5013])
netflix_LGN_hr_20 = np.array([0.8633,0.864,0.8643,0.858,0.855,0.837,0.8373,0.8373,0.8357])
netflix_LGN_ndcg_20 = np.array([0.2406,0.2371,0.2332,0.2288,0.2241,0.2132,0.213,0.2124,0.2117])

netflix_BPR_mia_acc = np.array([0.5083, 0.5083, 0.5083, 0.5085, 0.5082, 0.5084, 0.5085, 0.5085, 0.5085])
netflix_BPR_hr_20 = np.array([0.855,0.8533,0.8527,0.8487,0.8417,0.827,0.8257,0.8273,0.8263])
netflix_BPR_ndcg_20 = np.array([0.2364,0.234,0.2308,0.2263,0.2207,0.2119,0.2112,0.2111,0.2108])

# 配色
color_zrf = '#1f77b4'
color_rrecall = '#ff7f0e'
color_rndcg = '#2ca02c'

# 创建 1x2 子图
fig, axs = plt.subplots(1, 2, figsize=(9.8, 2.7), dpi=300, gridspec_kw={'wspace': 0.37})

def plot_dual_axis(ax, y1, y2, zrf_score, r_recall_20, r_ndcg_20, ylim):
    ax1 = ax

    # 左轴：ZRF
    ax1.plot(alpha, zrf_score, marker='o', linewidth=2, color=color_zrf, label='MIA ACC')
    # if y1:
    #     ax1.set_ylabel('Metric', fontsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x',  labelsize=15)

    ax1.set_ylim(ylim)
    ax1.set_xlabel('alpha', fontsize=15)
    ax1.grid(True)

    ax1.plot(alpha, r_recall_20, marker='s', linewidth=2, color=color_rrecall, label='R@10')
    ax1.plot(alpha, r_ndcg_20, marker='^', linewidth=2, color=color_rndcg, label='N@10')
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper left', ncol=3, fontsize=10)

# 画两个一模一样的图
plot_dual_axis(axs[0], True, False, ml1m_LGN_mia_acc, ml1m_LGN_hr_20, ml1m_LGN_ndcg_20, (0.65, 0.8))
plot_dual_axis(axs[1], True, False, ml1m_BPR_mia_acc, ml1m_BPR_hr_20, ml1m_BPR_ndcg_20, (0.65, 0.8))

# 设置标题
axs[0].set_title('ML-1M (LightGCN)', fontsize=15)
axs[1].set_title('Netflix (BPR)', fontsize=15)

plt.tight_layout()
plt.savefig("/Users/hebert/Desktop/alpha.pdf", bbox_inches='tight')
# plt.show()
