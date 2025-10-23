import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('/root/haichao/LLM_rec_unlearning')
from recbole_utils import RecUtils
import pandas as pd

# 获取candidate item 的传统推荐模型
MODEL = "LightGCN"
# 处理的数据集
DATASET = "netflix"
# 默认配置文件， 注意 normalize_all: False 便于保留原始的时间和rating
topK = [5, 10, 20]
config_files = f"config_file/{DATASET}.yaml"
config = {"normalize_all": False, "topk": topK}
config_file_list = (
    config_files.strip().split(" ") if config_files else None
)

rec_utils = RecUtils(model=MODEL, dataset=DATASET, config_file_list=config_file_list, config_dict=config)

def load_user_interactions(dataset_path):
    """
    从数据集文件加载用户的真实历史交互
    """
    user_interactions = {}
    
    with open(dataset_path, 'r') as f:
        lines = f.readlines()[1:]  # 跳过标题行
        
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            user_id = parts[0]
            item_id = parts[1]
            
            if user_id not in user_interactions:
                user_interactions[user_id] = set()
            user_interactions[user_id].add(item_id)
    
    return user_interactions

def load_recommendations(file_path):
    """
    加载推荐结果文件
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def convert_recommendations_to_encoded(recommendations):
    """
    将推荐结果转换为编码后的形式
    """
    encoded_recommendations = {}
    for user_id, items in recommendations.items():
        try:
            # 尝试获取编码后的用户ID
            encoded_user_id = rec_utils.get_encode_user_token(user_id)
            
            # 处理不同的推荐结果格式
            if isinstance(items, dict):
                # 新格式：{item_id: score, ...}
                encoded_items = {}
                for item_id, score in items.items():
                    try:
                        encoded_item_id = rec_utils.get_encode_item_token(item_id)
                        encoded_items[str(encoded_item_id)] = float(score)
                    except:
                        encoded_items[item_id] = float(score)  # 如果无法编码，保持原样
            else:
                # 旧格式：[item_id, ...]
                encoded_items = []
                for item_id in items:
                    try:
                        encoded_item_id = rec_utils.get_encode_item_token(item_id)
                        encoded_items.append(str(encoded_item_id))
                    except:
                        encoded_items.append(item_id)  # 如果无法编码，保持原样
                        
            encoded_recommendations[str(encoded_user_id)] = encoded_items
        except:
            # 如果无法编码用户，跳过该用户
            continue
    return encoded_recommendations

import numpy as np
import math

def extract_features(user_id, user_history, recommendation_list, k=50):
    """
    从推荐列表中提取特征，旨在对排名变化更敏感。
    :param user_id: 用户ID (str or int, for context, not used in calculation)
    :param user_history: 用户真实历史交互集合 (set of item IDs as strings)
    :param recommendation_list: Top-K 推荐物品ID列表 (list of item IDs as strings) 或 {item_id: score} 字典
    :param k: Top-K值
    :return: 特征向量 [命中数, 历史DCG, 首个命中项的倒数排名, 命中项排名的标准差, 最高命中排名, 平均分数, 分数方差]
    """
    # 处理推荐列表格式，统一转换为物品ID列表
    if isinstance(recommendation_list, dict):
        # 新格式：{item_id: score, ...}，按分数排序
        sorted_items = sorted(recommendation_list.items(), key=lambda x: x[1], reverse=True)
        item_ids = [item_id for item_id, score in sorted_items]
        scores = [score for item_id, score in sorted_items]
    else:
        # 旧格式：[item_id, ...]
        item_ids = recommendation_list
        scores = None  # 无分数信息
    
    # 1. Prepare list and identify hits
    original_list_len = len(item_ids)
    if original_list_len > k:
        item_ids = item_ids[:k]
        if scores is not None:
            scores = scores[:k]
    elif original_list_len < k:
        # Pad with a non-matching placeholder (e.g., '-1')
        item_ids.extend(['-1'] * (k - original_list_len))
        if scores is not None:
            scores.extend([0.0] * (k - original_list_len))

    hit_item_ranks = {} # Store item_id -> rank (1-based)
    ranks_of_hits = []
    hit_scores = []  # 存储命中项的分数
    
    for i, (item_id, score) in enumerate(zip(item_ids, scores if scores is not None else [None]*len(item_ids))):
        if item_id in user_history:
            rank = i + 1
            hit_item_ranks[item_id] = rank
            ranks_of_hits.append(rank)
            if scores is not None:
                hit_scores.append(score)  # 记录命中项的分数

    hit_count = len(ranks_of_hits)

    # 2. Calculate Features

    # Feature 1: 命中数量 (Hit Count @K) - Remains fundamental
    f_hit_count = float(hit_count)

    # Feature 2: 基于历史命中的DCG@K (History-Aware DCG@K)
    # This feature gives more weight to hits at higher ranks.
    f_history_dcg = 0.0
    if hit_count > 0:
        for rank in ranks_of_hits:
            # Standard DCG formula: gain=1 for hits, 0 otherwise
            f_history_dcg += 1.0 / math.log2(rank + 1)
            
    # Feature 3: 首个命中项的倒数排名 (Reciprocal Rank of First Hit)
    # This focuses specifically on the highest-ranked hit item. Sensitive to changes at the top.
    f_rr_first_hit = 0.0
    if hit_count > 0:
        min_rank = min(ranks_of_hits)
        f_rr_first_hit = 1.0 / min_rank

    # Feature 4: 命中项排名的标准差 (Standard Deviation of Hit Ranks)
    # Captures the spread/distribution of hit ranks. Might change if ranks shift significantly.
    f_std_dev_rank = 0.0
    if hit_count > 1: # Standard deviation requires at least 2 points
        f_std_dev_rank = np.std(ranks_of_hits)

    # Feature 5: 最高命中排名 (Min Rank of Hits) - Kept for comparison/potential signal
    f_min_rank = float(k + 1) # Default value if no hits
    if hit_count > 0:
        f_min_rank = float(min(ranks_of_hits))

    # Feature 6: 平均分数 (Average Score) - 新增特征
    f_avg_score = 0.0
    if scores is not None:
        f_avg_score = np.mean(scores)
    
    # Feature 7: 分数方差 (Score Variance) - 新增特征
    f_score_variance = 0.0
    if scores is not None and len(scores) > 1:
        f_score_variance = np.var(scores)
        
    # Feature 8: 命中项的平均分数 - 使用物品预测分数的重要特征
    f_hit_avg_score = 0.0
    if hit_scores:
        f_hit_avg_score = np.mean(hit_scores)
        
    # Feature 9: 命中项分数的标准差 - 使用物品预测分数的重要特征
    f_hit_score_std = 0.0
    if len(hit_scores) > 1:
        f_hit_score_std = np.std(hit_scores)

    # Return the chosen features
    # We choose features potentially more sensitive to rank changes:
    return [
        f_hit_count,        # How many historical items are recommended?
        f_history_dcg,      # Are the recommended historical items ranked high (weighted)?
        f_rr_first_hit,     # How high is the *highest ranked* historical item?
        f_std_dev_rank,     # Are the recommended historical items clustered or spread out in rank?
        f_min_rank,         # (Redundant with f_rr_first_hit but kept for now) What is the absolute highest rank?
        f_avg_score,        # 平均推荐分数
        f_score_variance,   # 推荐分数方差
        f_hit_avg_score,    # 命中项的平均分数
        f_hit_score_std     # 命中项分数的标准差
    ]

def prepare_training_data(retain_interactions, test_interactions, base_recommendations, k=50):
    """
    准备攻击模型的训练数据
    :param retain_interactions: 保留集用户交互（正样本）
    :param test_interactions: 测试集用户交互（负样本）
    :param base_recommendations: 基础模型推荐结果
    :param k: Top-K值
    :return: 特征矩阵X和标签向量y
    """
    # 随机抽样一部分用户用于训练
    retain_users = list(retain_interactions.keys())
    test_users = list(test_interactions.keys())
    
    # 抽样相同样本数量的用户
    sample_size = min(len(retain_users), len(test_users), 1000)  # 限制样本数量以提高效率
    sampled_retain_users = random.sample(retain_users, sample_size)
    sampled_test_users = random.sample(test_users, sample_size)
    
    X = []  # 特征
    y = []  # 标签: 1表示成员，0表示非成员
    
    # 处理正样本（保留集用户）
    for user_id in sampled_retain_users:
        if user_id in base_recommendations:
            user_history = retain_interactions[user_id]
            recommendation_list = base_recommendations[user_id]
            features = extract_features(user_id, user_history, recommendation_list, k)
            X.append(features)
            y.append(1)
    
    # 处理负样本（测试集用户）
    for user_id in sampled_test_users:
        if user_id in base_recommendations:
            user_history = test_interactions[user_id]
            recommendation_list = base_recommendations[user_id]
            features = extract_features(user_id, user_history, recommendation_list, k)
            X.append(features)
            y.append(0)
    
    return np.array(X), np.array(y)

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import torch # 假设输入可能是 torch tensors

def train_attack_model(X, y, test_size=0.2, tune_hyperparameters=True, random_state=42):
    """
    改进的训练攻击模型函数，包含验证集、超参数调优（可选）和性能报告。

    Args:
        X (np.ndarray or torch.Tensor): 攻击模型的完整特征数据 (members + non-members)。
        y (np.ndarray or torch.Tensor): 对应的标签 (1 for members, 0 for non-members)。
        test_size (float): 用于内部验证集的比例。
        tune_hyperparameters (bool): 是否执行超参数搜索。
        random_state (int): 用于可复现性。

    Returns:
        tuple: (best_clf, scaler, report)
               best_clf: 训练好的最佳分类器。
               scaler: 在训练集上拟合好的StandardScaler。
               report (dict): 包含训练和验证性能指标的报告。
    """
    # 确保数据是 numpy array
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    # --- 1. 划分训练集和内部验证集 ---
    X_attack_train, X_attack_val, y_attack_train, y_attack_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y # stratify 保证类别比例
    )
    print(f"攻击者训练数据: {len(X_attack_train)} 样本")
    print(f"攻击者验证数据: {len(X_attack_val)} 样本")

    # --- 2. 特征标准化 (仅在训练集上fit, 然后transform训练集和验证集) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_attack_train)
    X_val_scaled = scaler.transform(X_attack_val) # 使用相同的 scaler

    best_clf = None
    report = {}

    # --- 3. 模型训练与选择 ---
    if tune_hyperparameters:
        print("正在为逻辑回归执行超参数调优...")
        # 定义参数网格 (增加迭代次数以确保收敛)
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']} 
        
        # 使用GridSearchCV进行交叉验证调优
        grid_search = GridSearchCV(
            LogisticRegression(random_state=random_state, max_iter=100, class_weight='balanced'), # 添加 class_weight
            param_grid,
            cv=5, # 5折交叉验证
            scoring='accuracy', # 优化目标：准确率 (也可以用 'roc_auc')
            n_jobs=-1 # 使用所有CPU核心
        )
        grid_search.fit(X_train_scaled, y_attack_train)

        print(f"通过GridSearchCV找到的最佳超参数: {grid_search.best_params_}")
        best_clf = grid_search.best_estimator_ # 获取在整个训练集上用最佳参数训练好的模型
        report['best_params'] = grid_search.best_params_

    else:
        # 如果不调优，直接训练默认模型
        print("使用默认参数 (C=1.0) 训练逻辑回归...")
        # 同样增加迭代次数并平衡类别权重
        clf = LogisticRegression(random_state=random_state, max_iter=2000, class_weight='balanced')
        clf.fit(X_train_scaled, y_attack_train)
        best_clf = clf
        report['best_params'] = 'default (C=1.0)'

    # --- 4. 在训练集和验证集上评估最终模型 ---
    y_train_pred_proba = best_clf.predict_proba(X_train_scaled)[:, 1]
    y_val_pred_proba = best_clf.predict_proba(X_val_scaled)[:, 1]
    
    y_train_pred_label = (y_train_pred_proba > 0.5).astype(int)
    y_val_pred_label = (y_val_pred_proba > 0.5).astype(int)

    # 检查验证集标签是否单一，以防AUC计算报错
    val_auc = 0.5
    if len(np.unique(y_attack_val)) > 1:
        val_auc = roc_auc_score(y_attack_val, y_val_pred_proba)
        
    train_auc = 0.5
    if len(np.unique(y_attack_train)) > 1:
          train_auc = roc_auc_score(y_attack_train, y_train_pred_proba)

    report['train'] = {
        'accuracy': accuracy_score(y_attack_train, y_train_pred_label),
        'auc': train_auc
    }
    report['validation'] = {
        'accuracy': accuracy_score(y_attack_val, y_val_pred_label),
        'auc': val_auc
    }

    print("\n攻击者性能报告:")
    print(f"  训练集 Accuracy: {report['train']['accuracy']:.4f}")
    print(f"  训练集 AUC: {report['train']['auc']:.4f}")
    print(f"  验证集 Accuracy: {report['validation']['accuracy']:.4f}")
    print(f"  验证集 AUC: {report['validation']['auc']:.4f}")

    # 检查过拟合
    overfitting_threshold = 0.1 # 可以调整这个阈值
    if report['train']['accuracy'] > report['validation']['accuracy'] + overfitting_threshold:
        print("警告: 检测到潜在过拟合 (训练集ACC >> 验证集ACC)")
    if report['train']['auc'] > report['validation']['auc'] + overfitting_threshold:
         print("警告: 检测到潜在过拟合 (训练集AUC >> 验证集AUC)")

    return best_clf, scaler, report

def evaluate_model(clf, scaler, X_test, y_test):
    """
    评估模型性能
    """
    # 特征标准化
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]  # 正类的概率
    
    acc = accuracy_score(y_test, y_pred)
    # 只有当y_test中包含两个类别时才计算AUC
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = 0.5  # 当只有一类时，AUC无定义，设为0.5
    
    return acc, auc

def evaluate_forget_effectiveness(clf, scaler, forget_interactions, 
                                 recommendations_mbase, recommendations_munlearned, recommendations_mgold,
                                 encoded_forget_interactions, encoded_test_interactions, encoded_retain_interactions, k=50):
    """
    评估遗忘效果
    :param clf: 训练好的攻击模型
    :param scaler: 特征标准化器
    :param forget_interactions: 遗忘集用户交互（原始ID）
    :param recommendations_mbase: 基础模型推荐结果（编码后ID）
    :param recommendations_munlearned: 遗忘后模型推荐结果（编码后ID）
    :param recommendations_mgold: 重训模型推荐结果（编码后ID）
    :param encoded_forget_interactions: 编码后的遗忘集用户交互
    :param encoded_test_interactions: 编码后的测试集用户交互（用于负样本）
    :param encoded_retain_interactions: 编码后的保留集用户交互
    :param k: Top-K值
    :return: 各模型在遗忘集上的ACC和AUC
    """
    forget_users = list(forget_interactions.keys())
    encoded_forget_users = list(encoded_forget_interactions.keys())
    encoded_test_users = list(encoded_test_interactions.keys())
    encoded_retain_users = list(encoded_retain_interactions.keys())
    
    results = {}
    
    # 随机选择测试集用户作为负样本
    test_users_sample = random.sample(encoded_test_users, 
                                    min(len(encoded_forget_users), len(encoded_test_users)))
    
    # 随机选择部分retain_set用户用于AUC计算
    retain_users_sample = random.sample(encoded_retain_users, 
                                      min(len(encoded_forget_users), len(encoded_retain_users)))
    
    # 评估基础模型 (M_base)
    # ACC计算：forget_set (标签1) vs test_set (标签0)
    X_mbase_acc_positive = []
    for user_id in encoded_forget_users:
        if user_id in recommendations_mbase:
            user_history = encoded_forget_interactions[user_id]
            recommendation_list = recommendations_mbase[user_id]
            features = extract_features(user_id, user_history, recommendation_list, k)
            X_mbase_acc_positive.append(features)
    
    X_mbase_acc_negative = []
    for user_id in test_users_sample:
        if user_id in recommendations_mbase:
            user_history = encoded_test_interactions[user_id]
            recommendation_list = recommendations_mbase[user_id]
            features = extract_features(user_id, user_history, recommendation_list, k)
            X_mbase_acc_negative.append(features)
    
    if X_mbase_acc_positive and X_mbase_acc_negative:
        X_mbase_acc = np.array(X_mbase_acc_positive + X_mbase_acc_negative)
        X_mbase_acc_scaled = scaler.transform(X_mbase_acc)
        y_pred_proba_mbase_acc = clf.predict_proba(X_mbase_acc_scaled)[:, 1]
        # 正样本标签为1，负样本标签为0
        y_true_acc = [1] * len(X_mbase_acc_positive) + [0] * len(X_mbase_acc_negative)
        acc_mbase = accuracy_score(y_true_acc, (y_pred_proba_mbase_acc > 0.5).astype(int))
        print(f"M_base预测概率 - 平均值: {np.mean(y_pred_proba_mbase_acc):.4f}, 标准差: {np.std(y_pred_proba_mbase_acc):.4f}")
        print(f"M_base预测概率范围: [{np.min(y_pred_proba_mbase_acc):.4f}, {np.max(y_pred_proba_mbase_acc):.4f}]")
        
        # AUC计算：forget_set + 部分retain_set (标签1) vs test_set (标签0)
        X_mbase_auc_positive_forget = X_mbase_acc_positive  # forget_set用户
        
        X_mbase_auc_positive_retain = []
        for user_id in retain_users_sample:
            if user_id in recommendations_mbase:
                user_history = encoded_retain_interactions[user_id]
                recommendation_list = recommendations_mbase[user_id]
                features = extract_features(user_id, user_history, recommendation_list, k)
                X_mbase_auc_positive_retain.append(features)
        
        X_mbase_auc_positive = X_mbase_auc_positive_forget + X_mbase_auc_positive_retain
        X_mbase_auc_negative = X_mbase_acc_negative  # test_set用户
        
        if X_mbase_auc_positive and X_mbase_auc_negative:
            X_mbase_auc = np.array(X_mbase_auc_positive + X_mbase_auc_negative)
            X_mbase_auc_scaled = scaler.transform(X_mbase_auc)
            y_pred_proba_mbase_auc = clf.predict_proba(X_mbase_auc_scaled)[:, 1]
            # 正样本标签为1，负样本标签为0
            y_true_auc = [1] * len(X_mbase_auc_positive) + [0] * len(X_mbase_auc_negative)
            auc_mbase = roc_auc_score(y_true_auc, y_pred_proba_mbase_auc)
        else:
            auc_mbase = 0.5
            
        results['M_base'] = {'ACC': acc_mbase, 'AUC': auc_mbase, 'pred_mean': np.mean(y_pred_proba_mbase_acc)}
    
    # 评估遗忘后模型 (M_unlearned)
    # ACC计算：forget_set (标签1) vs test_set (标签0)
    X_munlearned_acc_positive = []
    for user_id in encoded_forget_users:
        if user_id in recommendations_munlearned:
            user_history = encoded_forget_interactions[user_id]
            recommendation_list = recommendations_munlearned[user_id]
            features = extract_features(user_id, user_history, recommendation_list, k)
            X_munlearned_acc_positive.append(features)
    
    X_munlearned_acc_negative = []
    for user_id in test_users_sample:
        if user_id in recommendations_munlearned:
            user_history = encoded_test_interactions[user_id]
            recommendation_list = recommendations_munlearned[user_id]
            features = extract_features(user_id, user_history, recommendation_list, k)
            X_munlearned_acc_negative.append(features)
    
    if X_munlearned_acc_positive and X_munlearned_acc_negative:
        X_munlearned_acc = np.array(X_munlearned_acc_positive + X_munlearned_acc_negative)
        X_munlearned_acc_scaled = scaler.transform(X_munlearned_acc)
        y_pred_proba_munlearned_acc = clf.predict_proba(X_munlearned_acc_scaled)[:, 1]
        # 正样本标签为1，负样本标签为0
        y_true_acc = [1] * len(X_munlearned_acc_positive) + [0] * len(X_munlearned_acc_negative)
        acc_munlearned = accuracy_score(y_true_acc, (y_pred_proba_munlearned_acc > 0.5).astype(int))
        print(f"M_unlearned预测概率 - 平均值: {np.mean(y_pred_proba_munlearned_acc):.4f}, 标准差: {np.std(y_pred_proba_munlearned_acc):.4f}")
        print(f"M_unlearned预测概率范围: [{np.min(y_pred_proba_munlearned_acc):.4f}, {np.max(y_pred_proba_munlearned_acc):.4f}]")
        
        # AUC计算：forget_set + 部分retain_set (标签1) vs test_set (标签0)
        X_munlearned_auc_positive_forget = X_munlearned_acc_positive  # forget_set用户
        
        X_munlearned_auc_positive_retain = []
        for user_id in retain_users_sample:
            if user_id in recommendations_munlearned:
                user_history = encoded_retain_interactions[user_id]
                recommendation_list = recommendations_munlearned[user_id]
                features = extract_features(user_id, user_history, recommendation_list, k)
                X_munlearned_auc_positive_retain.append(features)
        
        X_munlearned_auc_positive = X_munlearned_auc_positive_forget + X_munlearned_auc_positive_retain
        X_munlearned_auc_negative = X_munlearned_acc_negative  # test_set用户
        
        if X_munlearned_auc_positive and X_munlearned_auc_negative:
            X_munlearned_auc = np.array(X_munlearned_auc_positive + X_munlearned_auc_negative)
            X_munlearned_auc_scaled = scaler.transform(X_munlearned_auc)
            y_pred_proba_munlearned_auc = clf.predict_proba(X_munlearned_auc_scaled)[:, 1]
            # 正样本标签为1，负样本标签为0
            y_true_auc = [1] * len(X_munlearned_auc_positive) + [0] * len(X_munlearned_auc_negative)
            auc_munlearned = roc_auc_score(y_true_auc, y_pred_proba_munlearned_auc)
        else:
            auc_munlearned = 0.5
            
        results['M_unlearned'] = {'ACC': acc_munlearned, 'AUC': auc_munlearned, 'pred_mean': np.mean(y_pred_proba_munlearned_acc)}
    
    # 评估重训模型 (M_gold)
    # ACC和AUC计算：retain_set (标签1) vs forget_set + test_set (标签0)
    X_mgold_positive = []
    for user_id in retain_users_sample:
        if user_id in recommendations_mgold:
            user_history = encoded_retain_interactions[user_id]
            recommendation_list = recommendations_mgold[user_id]
            features = extract_features(user_id, user_history, recommendation_list, k)
            X_mgold_positive.append(features)
    
    # 负样本：forget set用户
    X_mgold_negative_forget = []
    for user_id in encoded_forget_users:
        if user_id in recommendations_mgold:
            user_history = encoded_forget_interactions[user_id]
            recommendation_list = recommendations_mgold[user_id]
            features = extract_features(user_id, user_history, recommendation_list, k)
            X_mgold_negative_forget.append(features)
    
    # 负样本：test set用户
    X_mgold_negative_test = []
    for user_id in test_users_sample:
        if user_id in recommendations_mgold:
            user_history = encoded_test_interactions[user_id]
            recommendation_list = recommendations_mgold[user_id]
            features = extract_features(user_id, user_history, recommendation_list, k)
            X_mgold_negative_test.append(features)
    
    # 合并负样本
    X_mgold_negative = X_mgold_negative_forget + X_mgold_negative_test
    
    if X_mgold_positive and X_mgold_negative:
        X_mgold = np.array(X_mgold_positive + X_mgold_negative)
        X_mgold_scaled = scaler.transform(X_mgold)
        y_pred_proba_mgold = clf.predict_proba(X_mgold_scaled)[:, 1]
        # 正样本标签为1，负样本标签为0
        y_true = [1] * len(X_mgold_positive) + [0] * len(X_mgold_negative)
        acc_mgold = accuracy_score(y_true, (y_pred_proba_mgold > 0.5).astype(int))
        auc_mgold = roc_auc_score(y_true, y_pred_proba_mgold)
        print(f"M_gold预测概率 - 平均值: {np.mean(y_pred_proba_mgold):.4f}, 标准差: {np.std(y_pred_proba_mgold):.4f}")
        print(f"M_gold预测概率范围: [{np.min(y_pred_proba_mgold):.4f}, {np.max(y_pred_proba_mgold):.4f}]")
        results['M_gold'] = {'ACC': acc_mgold, 'AUC': auc_mgold, 'pred_mean': np.mean(y_pred_proba_mgold)}
    elif X_mgold_positive or X_mgold_negative:
        # 即使只有一种样本，我们也计算结果
        X_mgold = np.array(X_mgold_positive if X_mgold_positive else X_mgold_negative)
        X_mgold_scaled = scaler.transform(X_mgold)
        y_pred_proba_mgold = clf.predict_proba(X_mgold_scaled)[:, 1]
        # 如果只有一种类型的样本，AUC无法计算
        y_true = [1] * len(X_mgold_positive) if X_mgold_positive else [0] * len(X_mgold_negative)
        acc_mgold = accuracy_score(y_true, (y_pred_proba_mgold > 0.5).astype(int)) if len(np.unique(y_true)) > 1 else 0.5
        auc_mgold = roc_auc_score(y_true, y_pred_proba_mgold) if len(np.unique(y_true)) > 1 else 0.5
        print(f"M_gold预测概率 - 平均值: {np.mean(y_pred_proba_mgold):.4f}, 标准差: {np.std(y_pred_proba_mgold):.4f}")
        print(f"M_gold预测概率范围: [{np.min(y_pred_proba_mgold):.4f}, {np.max(y_pred_proba_mgold):.4f}]")
        results['M_gold'] = {'ACC': acc_mgold, 'AUC': auc_mgold, 'pred_mean': np.mean(y_pred_proba_mgold)}
    
    return results

def convert_interactions_to_encoded(interactions):
    """
    将用户交互转换为编码后的形式
    """
    encoded_interactions = {}
    for user_id, items in interactions.items():
        try:
            # 尝试获取编码后的用户ID
            encoded_user_id = rec_utils.get_encode_user_token(user_id)
            encoded_items = set()
            for item_id in items:
                try:
                    encoded_item_id = rec_utils.get_encode_item_token(item_id)
                    encoded_items.add(str(encoded_item_id))
                except:
                    # 如果无法编码，跳过该项目
                    continue
            encoded_interactions[str(encoded_user_id)] = encoded_items
        except:
            # 如果无法编码用户，跳过该用户
            continue
    return encoded_interactions

def get_test_interactions_from_rec_utils():
    """
    从RecUtils中获取测试集用户交互作为负样本
    """
    test_interactions = {}
    
    # 获取测试集数据
    test_data = rec_utils.ori_testset
    
    # 按用户分组
    for _, row in test_data.iterrows():
        user_id = str(row['user_id'])
        item_id = str(row['item_id'])
        
        if user_id not in test_interactions:
            test_interactions[user_id] = set()
        test_interactions[user_id].add(item_id)
    
    return test_interactions

def main():
    # 文件路径
    base_rec_file = 'netflix_LightGCN_recommendations_top50.json'  # M_base推荐结果
    retrain_rec_file = 'netflix-remain_LightGCN_remainset_top50.json'  # M_gold推荐结果
    unlearned_rec_file = 'netflix_LightGCN_recommendations_top20_updated.json'  # M_unlearned推荐结果
    
    # 数据集文件路径
    retain_dataset_file = 'dataset/netflix-remain/netflix-remain.inter'  # 保留集
    forget_dataset_file = 'dataset/netflix-forget/netflix-forget.inter'  # 遗忘集
    
    print("正在加载数据...")
    
    # 加载用户交互数据
    retain_interactions = load_user_interactions(retain_dataset_file)
    forget_interactions = load_user_interactions(forget_dataset_file)
    
    print(f"保留集用户数: {len(retain_interactions)}")
    print(f"遗忘集用户数: {len(forget_interactions)}")
    
    # 从RecUtils获取测试集用户交互作为负样本
    print("正在获取测试集用户交互...")
    test_interactions = get_test_interactions_from_rec_utils()
    print(f"测试集用户数: {len(test_interactions)}")
    
    # 转换交互数据以匹配推荐结果的编码
    print("正在转换交互数据编码...")
    encoded_retain_interactions = convert_interactions_to_encoded(retain_interactions)
    encoded_forget_interactions = convert_interactions_to_encoded(forget_interactions)
    encoded_test_interactions = convert_interactions_to_encoded(test_interactions)
    
    print(f"编码后保留集用户数: {len(encoded_retain_interactions)}")
    print(f"编码后遗忘集用户数: {len(encoded_forget_interactions)}")
    print(f"编码后测试集用户数: {len(encoded_test_interactions)}")
    
    # 加载推荐结果
    base_recommendations = load_recommendations(base_rec_file)
    retrain_recommendations = load_recommendations(retrain_rec_file)
    unlearned_recommendations = load_recommendations(unlearned_rec_file)
    
    # 确保所有推荐结果中的用户ID和物品ID都是编码后的形式
    print("正在转换推荐结果编码...")
    encoded_base_recommendations = convert_recommendations_to_encoded(base_recommendations)
    encoded_retrain_recommendations = convert_recommendations_to_encoded(retrain_recommendations)
    encoded_unlearned_recommendations = convert_recommendations_to_encoded(unlearned_recommendations)
    
    print(f"编码后基础模型推荐用户数: {len(encoded_base_recommendations)}")
    print(f"编码后重训模型推荐用户数: {len(encoded_retrain_recommendations)}")
    print(f"编码后遗忘模型推荐用户数: {len(encoded_unlearned_recommendations)}")
    
    print("正在准备训练数据...")
    
    # 准备攻击模型训练数据
    X, y = prepare_training_data(encoded_retain_interactions, encoded_test_interactions, encoded_base_recommendations, k=50)
    
    # 分割训练和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"训练样本数: {len(X_train)}, 验证样本数: {len(X_val)}")
    print(f"训练集标签分布: {np.bincount(y_train)}")
    print(f"验证集标签分布: {np.bincount(y_val)}")
    
    # 显示特征统计信息
    print(f"特征维度: {X_train.shape[1]}")
    print(f"训练集特征统计:")
    feature_names = [
        "命中数量", "历史DCG", "首个命中项倒数排名", "命中项排名标准差", 
        "最高命中排名", "平均分数", "分数方差", "命中项平均分数", "命中项分数标准差"
    ]
    for i, (name, mean, std) in enumerate(zip(feature_names, np.mean(X_train, axis=0), np.std(X_train, axis=0))):
        print(f"  {name}: 均值={mean:.4f}, 标准差={std:.4f}")
    
    # 检查是否有无效值
    print(f"训练集中NaN值数量: {np.isnan(X_train).sum()}")
    print(f"训练集中无穷值数量: {np.isinf(X_train).sum()}")
    
    # 替换无效值
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=100.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=100.0, neginf=0.0)
    
    print("正在训练攻击模型...")
    
    # 训练攻击模型
    attack_model, scaler, r = train_attack_model(X_train, y_train)
    
    # 在验证集上评估
    val_acc, val_auc = evaluate_model(attack_model, scaler, X_val, y_val)
    print(f"攻击模型在验证集上的性能: ACC={val_acc:.4f}, AUC={val_auc:.4f}")
    
    print("正在评估遗忘效果...")
    
    # 评估遗忘效果
    forget_results = evaluate_forget_effectiveness(
        attack_model, scaler, forget_interactions,
        encoded_base_recommendations, encoded_unlearned_recommendations, encoded_retrain_recommendations,
        encoded_forget_interactions, encoded_test_interactions, encoded_retain_interactions, k=50
    )
    
    print("\n遗忘效果评估结果:")
    print("=" * 50)
    for model_name, metrics in forget_results.items():
        print(f"{model_name}:")
        print(f"  ACC: {metrics['ACC']:.4f}")
        print(f"  AUC: {metrics['AUC']:.4f}")
        print(f"  平均预测概率: {metrics['pred_mean']:.4f}")
        print()
    
    # 计算遗忘效果（与基础模型相比的下降程度）
    if 'M_base' in forget_results and 'M_unlearned' in forget_results:
        acc_drop = forget_results['M_base']['ACC'] - forget_results['M_unlearned']['ACC']
        auc_drop = forget_results['M_base']['AUC'] - forget_results['M_unlearned']['AUC']
        pred_drop = forget_results['M_base']['pred_mean'] - forget_results['M_unlearned']['pred_mean']
        print(f"遗忘效果 (与M_base相比):")
        print(f"  ACC下降: {acc_drop:.4f}")
        print(f"  AUC下降: {auc_drop:.4f}")
        print(f"  平均预测概率下降: {pred_drop:.4f}")
    
    if 'M_base' in forget_results and 'M_gold' in forget_results:
        acc_drop_gold = forget_results['M_base']['ACC'] - forget_results['M_gold']['ACC']
        pred_drop_gold = forget_results['M_base']['pred_mean'] - forget_results['M_gold']['pred_mean']
        print(f"黄金标准 (与M_base相比):")
        print(f"  ACC下降: {acc_drop_gold:.4f}")
        print(f"  平均预测概率下降: {pred_drop_gold:.4f}")

if __name__ == "__main__":
    main()