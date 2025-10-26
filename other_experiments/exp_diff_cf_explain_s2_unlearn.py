import os
import time
from recbole_utils import RecUtils
import pandas as pd
import json
from tqdm import tqdm
from recbole.utils import set_color
from enum import Enum
import numpy as np
import torch
import json
import pandas as pd
from recbole_utils import RecUtils
import os
import numpy as np

# 从通用配置文件导入配置参数
from config import MODEL, DATASET, TOPK 
# 创建结果文件夹
import os
results_dir = f"{DATASET}_{MODEL}_diff_cf_experiment_results"
os.makedirs(results_dir, exist_ok=True)


# 处理的数据集
# DATASET = "ml-100k"  
# TOPK=20
alpha=1
# 默认配置文件， 注意 normalize_all: False 便于保留原始的时间和rating
config_files = f"config_file/{DATASET}.yaml"
config = {"normalize_all": False}
config_file_list = (
    config_files.strip().split(" ") if config_files else None
)
EXPLANATION_STRATEGY = "perturb-only"  # 可选: "CEU", "random", "grad-only", "perturb-only"

file_path=f"{results_dir}/{DATASET}_{MODEL}_all_counterfactual_explanations_{EXPLANATION_STRATEGY}.json"
# 保存调整后的推荐结果
output_file = f"{results_dir}/{DATASET}_{MODEL}_adjusted_recommendations_after_unlearning_{EXPLANATION_STRATEGY}.json"
# 保存更新后的推荐结果到新文件
updated_output_file = f"{results_dir}/{DATASET}_{MODEL}_recommendations_top{TOPK}_{EXPLANATION_STRATEGY}.json"

# 初始化RecUtils
rec_utils = RecUtils(model=MODEL, dataset=DATASET, config_file_list=config_file_list, config_dict=config)

def find_model_files(directory_path, model_name):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory_path):
        # 检查文件名是否包含 "abc"
        if model_name in filename and DATASET in filename:
            return os.path.join(directory_path, filename)

    return None

# 查找模型文件
model_file = find_model_files(directory_path=rec_utils.config["checkpoint_dir"], model_name=MODEL)
print(f"使用的模型文件: {model_file}")

# 加载推荐结果
def load_recommendations(file_path):
    """加载推荐结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 查找推荐结果文件
rec_file = f"{DATASET}_{MODEL}_recommendations_top50.json"
# 检查文件是否存在，如果不存在尝试其他可能的文件
if not os.path.exists(rec_file):
    possible_files = [f for f in os.listdir(".") if f.endswith("_recommendations_top50.json")]
    if possible_files:
        rec_file = possible_files[0]

if not os.path.exists(rec_file):
    raise FileNotFoundError(f"未找到推荐结果文件: {rec_file}")

# 加载推荐结果
recommendations = load_recommendations(rec_file)
print(f"加载推荐结果文件: {rec_file}")

def get_user_topk_recommendations(user_id_str, topk=50):
    """
    获取用户的Top-K推荐列表
    
    Args:
        user_id_str: 用户ID
        topk: 推荐数量
        
    Returns:
        list: 推荐物品列表（已按分数从高到低排序）
    """
    if user_id_str in recommendations:
        # 如果recommendations包含分数信息（字典格式），则提取物品ID
        user_recs = recommendations[user_id_str]
        if isinstance(user_recs, dict):
            # 新格式：{item_id: score, ...}
            # 按分数排序并提取前topk*2个物品ID
            sorted_items = sorted(user_recs.items(), key=lambda x: x[1], reverse=True)
            # 返回字典格式以保留分数信息
            return dict(sorted_items[:topk*2])
        else:
            # 旧格式：[item_id, ...]
            # 为了支持递补机制，我们返回两倍长度的推荐列表
            return user_recs[:topk*2]
    return []

def load_explanations():
    """
    加载预先生成的反事实解释
    
    Args:
        file_path: 解释文件路径
        
    Returns:
        dict: 解释数据
    """
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# 加载反事实解释数据
explanations_data = load_explanations()
print(f"加载反事实解释数据: {'成功' if explanations_data else '未找到文件'}")

def get_explanation_for_user_item(user_id_str, target_item_str, k=5):
    """
    获取指定用户和物品的反事实解释
    
    Args:
        user_id_str: 用户ID
        target_item_str: 目标物品ID
        k: 返回前k个最重要的解释
        
    Returns:
        list: 解释结果
    """
    # 从预先生成的解释数据中获取
    if user_id_str in explanations_data and target_item_str in explanations_data[user_id_str]:
        explanations = explanations_data[user_id_str][target_item_str]
        # 按重要性排序并返回前k个
        explanations.sort(key=lambda x: x['importance'], reverse=True)
        return explanations[:k]
    return []

def unlearning_process_single(user_id_str, forget_item_id_str, current_recommendations, topk=50):
    """
    执行单次遗忘学习过程，基于原始推荐分数进行调整
    
    Args:
        user_id_str: 用户ID
        forget_item_id_str: 需要遗忘的物品ID
        current_recommendations: 当前推荐列表（可能是物品ID列表或{item_id: score}字典）
        topk: 推荐列表长度
        
    Returns:
        list or dict: 调整后的推荐列表（如果是字典输入则返回字典，否则返回列表）
    """


    # 新格式：{item_id: score, ...}，按分数排序
    sorted_items = sorted(current_recommendations.items(), key=lambda x: x[1], reverse=True)
    # 不进行切片操作，处理所有推荐项
    all_recommendations = [(item_id, score) for item_id, score in sorted_items]

    # 一次性获取所有推荐物品的解释
    all_explanations = {}
    for item_id, score in all_recommendations:
        explanations = get_explanation_for_user_item(user_id_str, item_id, k=10)
        all_explanations[item_id] = explanations
    
    # 为每个推荐物品根据解释调整分数
    adjusted_scores = []
    
    for item_id, original_score in all_recommendations:
        # 从已获取的解释中查找
        explanations = all_explanations.get(item_id, [])
        
        # 检查被遗忘物品是否在解释中
        contribution_weight = 0.0
        for explanation in explanations:
            if explanation['item_id'] == forget_item_id_str:
                contribution_weight = explanation['importance']
                break
        
        # 根据贡献度调整分数
        adjusted_score = original_score
        if contribution_weight > 0:
            # 分数下降 importance * 2
            adjusted_score = original_score - (contribution_weight * alpha)
            # if user_id_str == '744':
                # print(f"{user_id_str} 的 {item_id} 的原始分数: {original_score}, 调整后分数: {adjusted_score}")

        adjusted_scores.append((item_id, adjusted_score))
    
    # 基于调整后的分数重新排序
    # 按分数降序排序
    adjusted_scores.sort(key=lambda x: x[1], reverse=True)
    

    adjusted_recommendations = {item_id: score for item_id, score in adjusted_scores}

    return adjusted_recommendations

def batch_unlearning_process(forget_set, topk=50):
    """
    批量处理遗忘学习过程
    
    Args:
        forget_set: 需要遗忘的数据集
        topk: 推荐列表长度
        
    Returns:
        dict: 所有用户的调整后推荐结果
    """
    # 按用户分组
    forget_users = forget_set['user_id'].unique()
    
    # 存储所有用户的调整后推荐结果
    adjusted_recommendations_all = {}
    
    print(f"\n开始批量处理 {len(forget_users)} 个用户的遗忘学习过程")
    
    for user_id in tqdm(forget_users, desc="处理用户遗忘请求"):
        # 获取该用户需要遗忘的所有物品
        user_forget_items = forget_set[forget_set['user_id'] == user_id]['item_id'].tolist()
        
        if user_forget_items:
            # 初始化调整后的推荐列表为原始推荐列表（不进行切片）
            current_recommendations = get_user_topk_recommendations(str(user_id), topk)
            
            # 对于每个需要遗忘的物品，依次调整推荐列表
            for i, forget_item in enumerate(user_forget_items):
                # 执行单次遗忘学习过程
                adjusted_rec = unlearning_process_single(str(user_id), str(forget_item), current_recommendations, topk)
                # 更新当前推荐列表，用于下一次迭代
                current_recommendations = adjusted_rec
            

            # 对于字典格式，按分数排序后取前K个
            sorted_items = sorted(current_recommendations.items(), key=lambda x: x[1], reverse=True)
            adjusted_recommendations_all[user_id] = {
                'adjusted_recommendations': [item_id for item_id, score in sorted_items[:topk]],  # 只在最终结果中取Top-K
                'adjusted_recommendations_with_scores': dict(sorted_items[:topk]),  # 保存调整后的分数
                'forget_items': user_forget_items,
                'forget_items_count': len(user_forget_items)
            }

    
    return adjusted_recommendations_all


def split_trainset_for_unlearning_by_interactions(trainset, forget_ratio=0.1, random_state=42):
    """
    将训练集按交互记录随机划分为Forget set和Remain set（不按用户划分）
    
    Args:
        trainset: 原始训练集DataFrame
        forget_ratio: 需要遗忘的数据比例，默认为0.1(10%)
        random_state: 随机种子，确保结果可重现
        
    Returns:
        forget_set: 需要遗忘的数据集
        remain_set: 剩余需要保留的数据集
    """
    # 设置随机种子以确保结果可重现
    np.random.seed(random_state)
    
    # 获取所有交互记录的索引
    all_indices = trainset.index.tolist()
    
    # 随机选择指定比例的交互记录作为forget set
    num_forget_interactions = max(1, int(len(all_indices) * forget_ratio))
    forget_indices = np.random.choice(all_indices, size=num_forget_interactions, replace=False)
    
    # 创建布尔索引
    forget_mask = trainset.index.isin(forget_indices)
    
    # 划分数据集
    forget_set = trainset[forget_mask].copy()
    remain_set = trainset[~forget_mask].copy()
    
    return forget_set, remain_set

# 执行划分
# 使用按用户划分的方法
# forget_set, remain_set = split_trainset_for_unlearning(rec_utils.ori_trainset)

# 使用按交互记录随机划分的方法
forget_set, remain_set = split_trainset_for_unlearning_by_interactions(rec_utils.ori_trainset)

# 示例：对forget用户执行批量遗忘学习过程
print(f"\n开始执行遗忘学习过程...")
start_time = time.time()
adjusted_results = batch_unlearning_process(forget_set, topk=TOPK)

end_time = time.time()
elapsed_time = end_time - start_time


with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(adjusted_results, f, ensure_ascii=False, indent=2)

print(f"\n调整后的推荐结果已保存到 {output_file}")
print(f"遗忘学习过程耗时: {elapsed_time:.2f} 秒")

# 创建新的推荐结果字典，将调整后的结果替换原始结果
updated_recommendations = recommendations.copy()  # 复制原始推荐结果

# 将调整后的推荐结果替换到新的推荐结果字典中
for user_id, result in adjusted_results.items():
    adjusted_recs = result['adjusted_recommendations']
    # 如果原始推荐结果包含分数信息，则保持相同的格式
    if isinstance(recommendations.get(str(user_id)), dict):
        # 使用调整后的分数信息
        adjusted_scores = result.get('adjusted_recommendations_with_scores', {})
        # 确保只包含在调整后推荐列表中的物品
        user_scores = {item_id: score for item_id, score in adjusted_scores.items() 
                      if item_id in adjusted_recs}
        # 按照分数倒排产生最终排名
        sorted_user_scores = dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True))
        
        updated_recommendations[str(user_id)] = sorted_user_scores
    else:
        # 保持原始格式
        updated_recommendations[str(user_id)] = adjusted_recs



with open(updated_output_file, 'w', encoding='utf-8') as f:
    json.dump(updated_recommendations, f, ensure_ascii=False, indent=2)

print(f"\n更新后的推荐结果已保存到 {updated_output_file}")
