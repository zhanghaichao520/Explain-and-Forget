import torch
import json
import pandas as pd
from recbole_utils import RecUtils
import os
import numpy as np
# 配置参数
MODEL = "BPR"
DATASET = "ml-100k" 
topK = 50
config_files = f"config_file/{DATASET}.yaml"
config = {"normalize_all": False}
config_file_list = config_files.strip().split(" ") if config_files else None

def load_recommendations(file_path):
    """加载推荐结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_user_history(rec_utils, user_id_str):
    """获取用户的历史交互记录"""
    # 获取用户的所有历史交互
    user_interactions = rec_utils.ori_trainset[rec_utils.ori_trainset['user_id'] == user_id_str]
    return user_interactions

class EmbeddingBasedCounterfactualExplainer:
    """
    基于嵌入的反事实解释生成器
    
    核心思想：在嵌入空间中进行反事实干预，通过优化扰动向量来最小化推荐分数
    """
    
    def __init__(self, rec_utils, model_file):
        self.rec_utils = rec_utils
        self.model_file = model_file
        # 加载模型
        self._load_model()
        # 提取嵌入向量
        self._extract_embeddings()
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(self.model_file, weights_only=False, map_location=self.rec_utils.config["device"])
            self.rec_utils.model.load_state_dict(checkpoint["state_dict"])
            if "other_parameter" in checkpoint:
                self.rec_utils.model.load_other_parameter(checkpoint.get("other_parameter"))
            self.rec_utils.model.eval()
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e
    
    def _extract_embeddings(self):
        """提取所有用户和物品的最终嵌入向量"""
        print("正在提取嵌入向量...")
        try:
            # 获取嵌入层
            self.user_embeddings = None
            self.item_embeddings = None
            
            # 尝试不同的属性名
            if hasattr(self.rec_utils.model, 'user_embedding'):
                self.user_embeddings = self.rec_utils.model.user_embedding.weight.data.clone()
            elif hasattr(self.rec_utils.model, 'embedding_dict'):
                if 'user' in self.rec_utils.model.embedding_dict:
                    self.user_embeddings = self.rec_utils.model.embedding_dict['user'].weight.data.clone()
            
            if hasattr(self.rec_utils.model, 'item_embedding'):
                self.item_embeddings = self.rec_utils.model.item_embedding.weight.data.clone()
            elif hasattr(self.rec_utils.model, 'embedding_dict'):
                if 'item' in self.rec_utils.model.embedding_dict:
                    self.item_embeddings = self.rec_utils.model.embedding_dict['item'].weight.data.clone()
            
            if self.user_embeddings is not None:
                print(f"用户嵌入维度: {self.user_embeddings.shape}")
            if self.item_embeddings is not None:
                print(f"物品嵌入维度: {self.item_embeddings.shape}")
                
        except Exception as e:
            print(f"提取嵌入向量时出错: {e}")
            raise e
    
    def compute_original_score(self, user_id_token, item_id_token):
        """计算用户对目标物品的原始推荐分数"""
        try:
            # 使用模型的full_sort_predict方法计算分数
            interaction = {self.rec_utils.dataset.uid_field: torch.tensor([user_id_token]).to(self.rec_utils.config["device"])}
            with torch.no_grad():
                scores = self.rec_utils.model.full_sort_predict(interaction)
                # LightGCN的full_sort_predict返回一维张量，包含用户对所有物品的评分
                # 直接通过索引获取目标物品的评分
                return scores[item_id_token].item()
        except Exception as e:
            print(f"计算原始分数时出错: {e}")
            # 备用方法：直接计算嵌入点积
            if self.user_embeddings is not None and self.item_embeddings is not None:
                user_emb = self.user_embeddings[user_id_token]
                item_emb = self.item_embeddings[item_id_token]
                return torch.dot(user_emb, item_emb).item()
            return 0.0
    
    def get_user_history_items(self, user_id_str):
        """获取用户历史交互物品ID列表"""
        # 直接从原始训练数据集中获取用户历史交互
        user_history = self.rec_utils.ori_trainset[self.rec_utils.ori_trainset['user_id'] == user_id_str]
        
        # 提取物品ID列表
        history_item_ids = user_history['item_id'].tolist()
        
        return history_item_ids
    
    def optimize_perturbation(self, user_id_token, item_id_token, user_id_str, iterations=50, lr=0.1):
        """
        优化扰动向量以最小化推荐分数
        
        Args:
            user_id_token: 用户token ID
            item_id_token: 目标物品token ID
            user_id_str: 用户ID字符串
            iterations: 优化迭代次数
            lr: 学习率
            
        Returns:
            tuple: (最优扰动向量, 优化过程记录)
        """
        
        # 获取用户历史交互物品
        history_item_ids = self.get_user_history_items(user_id_str)
        if not history_item_ids:
            raise ValueError("无法获取用户历史交互物品")
        
        # print(f"用户历史交互数量: {len(history_item_ids)}")
        
        # 获取用户和物品的嵌入向量
        if self.user_embeddings is None or self.item_embeddings is None:
            raise ValueError("嵌入向量未正确加载")
        
        user_embedding = self.user_embeddings[user_id_token]
        item_embedding = self.item_embeddings[item_id_token]
        
        # 获取原始分数
        original_score = self.compute_original_score(user_id_token, item_id_token)
        # print(f"原始推荐分数: {original_score:.4f}")
        
        # 获取历史物品的嵌入向量
        history_embeddings = []
        for item_id_str in history_item_ids:
            item_id_token_local = self.rec_utils.get_encode_item_token(item_id_str)
            history_embeddings.append(self.item_embeddings[item_id_token_local])
        
        history_embeddings = torch.stack(history_embeddings)
        
        # 初始化扰动向量（与历史交互数量相同）
        # 确保创建的是叶子张量
        perturbation = torch.randn(len(history_item_ids), device=self.rec_utils.config["device"]) * 0.1
        perturbation.requires_grad_(True)
        
        # 优化器 - 使用更强的优化器
        optimizer = torch.optim.Adam([perturbation], lr=lr)
        
        # 记录优化过程
        optimization_history = {
            'losses': [],
            'scores': [],
            'perturbation_norms': []
        }
        
        # 优化过程
        for i in range(iterations):
            # 重置梯度
            optimizer.zero_grad()
            
            # 修正后的逻辑：模拟LightGCN的聚合
            # 1. 将原始扰动向量映射到 [0, 1] 的掩码 P
            #    注意：sigmoid操作本身是可微分的，可以让梯度能够回传
            #    但我们希望扰动向量能够产生负值，因此我们调整sigmoid的中心点
            #    使用 sigmoid(perturbation) 将 (-inf, inf) -> (0, 1)
            #    但我们希望扰动向量为负时，掩码P接近0，扰动向量为正时，掩码P接近1
            final_mask_P = torch.sigmoid(perturbation) # Sigmoid 将 (-inf, inf) -> (0, 1)

            # 2. 模拟LightGCN的聚合
            #    获取用户自身的0阶嵌入（通常是模型初始化时的嵌入）
            #    为简化，我们这里仍然使用最终的用户嵌入作为基础，但干预其邻居贡献
            base_user_embedding = user_embedding 

            # 3. 聚合被扰动后的邻居信息
            #    history_embeddings 是一个 (num_history, embedding_dim) 的张量
            #    final_mask_P.unsqueeze(1) 将掩码扩展为 (num_history, 1)
            #    我们希望当扰动向量为负值时，对应的邻居贡献被抑制（接近0）
            #    当扰动向量为正值时，对应的邻居贡献被保留（接近1）
            weighted_history_embeddings = history_embeddings * final_mask_P.unsqueeze(1)
            aggregated_neighbors_embedding = torch.mean(weighted_history_embeddings, dim=0) # 使用均值聚合

            # 4. 得到新的用户嵌入（简化版聚合）
            perturbed_user_embedding = base_user_embedding + aggregated_neighbors_embedding
            
            # 计算扰动后的推荐分数 (点积)
            perturbed_score = torch.dot(perturbed_user_embedding, item_embedding)
            
            # 修正后的损失函数
            # 4.1 推荐翻转损失 (Flip Loss)
            #    目标是最小化扰动后的分数，我们希望它尽可能小（负值）
            flip_loss = perturbed_score

            # 4.2 稀疏性损失 (Sparsity Loss)
            #     目标是让掩码P尽可能接近1，即不扰动
            #     我们对(1 - P)进行L1惩罚，鼓励P接近1
            sparsity_loss = torch.norm(1.0 - final_mask_P, p=1)

            # 4.3 增加一个鼓励扰动向量产生区分度的损失项
            #     我们希望扰动向量中有明显的正负区分，而不是全部趋同
            #     通过最大化扰动向量的标准差来实现
            perturbation_std = torch.std(perturbation)
            diversity_loss = -perturbation_std * 50.0  # 负号表示最大化标准差，增加权重
            
            # 4.4 增加一个鼓励产生正扰动的损失项
            #     我们希望一部分扰动值能够变成正数，这样重要性得分就能产生区分度
            #     通过惩罚过大的负扰动值来实现
            large_negative_penalty = torch.sum(torch.relu(-perturbation - 1.0)) * 2.0

            # 总损失
            lambda_sparsity = 1 # 降低稀疏性损失的权重
            lambda_diversity = 0.5 # 降低多样性损失的权重
            lambda_penalty = 1.0 # 增加正扰动鼓励损失的权重
            total_loss = flip_loss + lambda_sparsity * sparsity_loss + lambda_diversity * diversity_loss + lambda_penalty * large_negative_penalty
            
            # 反向传播
            total_loss.backward()
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_value_([perturbation], 10.0)
            optimizer.step()
            
            # 记录过程
            optimization_history['losses'].append(total_loss.item())
            optimization_history['scores'].append(perturbed_score.item())
            optimization_history['perturbation_norms'].append(sparsity_loss.item())
            
            # 打印进度
            # if i % 100 == 0:
            #     print(f"迭代 {i}: 总损失 = {total_loss.item():.4f}, "
            #           f"扰动后分数 = {perturbed_score.item():.4f}, "
            #           f"扰动范数 = {sparsity_loss.item():.4f}")
            #     # 显示一些扰动值示例
            #     if len(perturbation) >= 5:
            #         sample_perturbations = perturbation[:5].detach().cpu().numpy()
            #         print(f"        扰动示例: {sample_perturbations}")
            #         # 显示对应的掩码P值示例
            #         sample_masks = final_mask_P[:5].detach().cpu().numpy()
            #         print(f"        掩码P示例: {sample_masks}")
        
        
        return perturbation.detach(), optimization_history, history_item_ids
    
    def explain(self, user_id_str, target_item_str, k=5):
        """
        生成基于嵌入的反事实解释
        
        Args:
            user_id_str: 用户ID
            target_item_str: 目标物品ID
            k: 返回前k个最重要的解释
            
        Returns:
            list: 解释结果
        """
        try:
            # 获取token ID
            user_id_token = self.rec_utils.get_encode_user_token(user_id_str)
            target_item_token = self.rec_utils.get_encode_item_token(target_item_str)
            
            # 优化扰动向量
            perturbation, optimization_history, history_item_ids = self.optimize_perturbation(
                user_id_token, target_item_token, user_id_str
            )
            
            # 获取用户历史交互（从原始训练数据中）
            user_history = get_user_history(self.rec_utils, user_id_str)
            
            # 创建物品ID到历史记录的映射
            item_to_history = {}
            for idx, row in user_history.iterrows():
                item_to_history[row['item_id']] = row

            # 1. 将原始扰动向量（可正可负）映射到 [0, 1] 的掩码 P
            #    使用 sigmoid 函数将扰动向量映射到 (0, 1) 区间
            #    这样可以正确反映扰动向量的抑制程度
            final_mask_P = torch.sigmoid(perturbation)
            
            # 2. 计算贡献度/重要性
            #    P值越接近0，说明被抑制得越厉害，贡献度越高。
            contribution_scores = 1.0 - final_mask_P
            
            importance_scores = []
            for i, item_id_str in enumerate(history_item_ids):
                if item_id_str in item_to_history:
                    row = item_to_history[item_id_str]
                    # 直接使用贡献度作为重要性
                    importance = contribution_scores[i].item()
                    
                    importance_scores.append({
                        'user_id': row['user_id'],
                        'item_id': row['item_id'],
                        'rating': row['rating'],
                        'importance': importance, # 这是修正后的重要性
                        'perturbation': perturbation[i].item(), # 仍然可以记录原始值用于调试
                        'index': i
                    })
            
            # 调试信息：显示前几个正数和负数扰动值的重要性得分

            positive_perturbations = [(i, perturbation[i].item(), contribution_scores[i].item()) for i in range(len(perturbation)) if perturbation[i].item() > 0]
            positive_perturbations.sort(key=lambda x: x[1], reverse=True)  # 按扰动值排序

            
            negative_perturbations = [(i, perturbation[i].item(), contribution_scores[i].item()) for i in range(len(perturbation)) if perturbation[i].item() < 0]
            negative_perturbations.sort(key=lambda x: x[1])  # 按扰动值排序

            
            # 按重要性排序（重要性越高，说明该交互越关键）
            importance_scores.sort(key=lambda x: x['importance'], reverse=True)
            
            # 构造解释结果
            explanation_results = []
            for i, exp in enumerate(importance_scores[:k], 1):
                explanation_results.append({ 
                    'item_id': exp['item_id'],
                    'importance': exp['importance']
                    # 'perturbation': exp['perturbation']   
                })
            
            return explanation_results
        except Exception as e:
            print(f"生成解释时出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    

def save_explanations_to_file(explanations, user_id, target_item, output_file=None):
    """
    将解释结果保存到文件
    
    Args:
        explanations: 解释结果列表
        user_id: 用户ID
        target_item: 目标物品ID
        output_file: 输出文件路径
    """
    if output_file is None:
        output_file = f"explanation_user_{user_id}_item_{target_item}.json"
    
    # 准备保存的数据
    data_to_save = {
        'user_id': user_id,
        'target_item': target_item,
        'explanations': explanations
    }
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    
    print(f"解释结果已保存到 {output_file}")

def save_all_explanations_to_file(all_explanations, output_file=f"{DATASET}_{MODEL}_all_counterfactual_explanations.json"):
    """
    将所有用户的解释结果保存到一个文件
    
    Args:
        all_explanations: 所有用户的解释结果字典
        output_file: 输出文件路径
    """
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_explanations, f, ensure_ascii=False, indent=2)
    
    print(f"所有解释结果已保存到 {output_file}")

def find_model_file(rec_utils, model, dataset):
    """查找模型文件"""
    model_file = None
    
    # 首先在checkpoint_dir中查找
    checkpoint_dir = rec_utils.config["checkpoint_dir"]
    if os.path.exists(checkpoint_dir):
        for filename in os.listdir(checkpoint_dir):
            if model in filename and dataset in filename:
                model_file = os.path.join(checkpoint_dir, filename)
                break
    
    # 如果没找到，尝试在saved目录下查找
    if model_file is None:
        saved_dir = "saved"
        if os.path.exists(saved_dir):
            for filename in os.listdir(saved_dir):
                if model in filename and dataset in filename:
                    model_file = os.path.join(saved_dir, filename)
                    break
    
    # 如果还是没找到，使用默认模型文件
    if model_file is None:
        saved_files = [f for f in os.listdir("saved") if f.endswith(".pth")]
        if saved_files:
            # 优先选择匹配数据集的模型
            for filename in saved_files:
                if dataset in filename:
                    model_file = os.path.join("saved", filename)
                    break
            # 如果没有匹配的，选择第一个
            if model_file is None:
                model_file = os.path.join("saved", saved_files[0])
    
    return model_file

from tqdm import tqdm
def main():

    # 初始化RecUtils
    rec_utils = RecUtils(model=MODEL, dataset=DATASET, config_file_list=config_file_list, config_dict=config)
    
    # 查找模型文件
    model_file = find_model_file(rec_utils, MODEL, DATASET)
    
    if model_file is None or not os.path.exists(model_file):
        raise FileNotFoundError(f"未找到匹配的模型文件: {model_file}")
    
    # 查找推荐结果文件
    rec_file = f"{DATASET}_{MODEL}_recommendations_top{topK}.json"
    
    if not os.path.exists(rec_file):
        raise FileNotFoundError(f"未找到推荐结果文件: {rec_file}")
    
    # 加载推荐结果
    recommendations = load_recommendations(rec_file)
    user_ids = list(recommendations.keys())
    
    # user_ids = user_ids[:2]
    if not user_ids:
        raise ValueError("推荐结果为空")
    
    print(f"=== 基于嵌入的反事实解释生成 ===")
    print(f"总共需要为 {len(user_ids)} 个用户生成解释")
    print("=" * 50)
    
    try:
        # 创建基于嵌入的反事实解释器
        explainer = EmbeddingBasedCounterfactualExplainer(rec_utils, model_file)
        
        # 存储所有用户的解释结果
        all_explanations = {}
        
        # 为每个用户生成解释
        for user_idx, user_id_str in enumerate(user_ids):
            print(f"\n处理用户 {user_id_str} ({user_idx+1}/{len(user_ids)})")
            
            # 获取该用户的所有推荐项
            recommended_items = recommendations[user_id_str]
            
            if not recommended_items:
                print(f"  用户 {user_id_str} 的推荐列表为空")
                continue
            
            # 为该用户的所有推荐项生成解释
            user_explanations = {}
            for item_idx, target_item_str in tqdm(enumerate(recommended_items), total=len(recommended_items)):
                # print(f"  处理推荐项 {target_item_str} ({item_idx+1}/{len(recommended_items)})")
                
                try:
                    # 生成反事实解释
                    explanations = explainer.explain(user_id_str, target_item_str, k=5)
                    
                    # 存储解释结果
                    user_explanations[target_item_str] = explanations
                    
                except Exception as e:
                    print(f"  为用户 {user_id_str} 的推荐项 {target_item_str} 生成解释时出错: {e}")
                    user_explanations[target_item_str] = []
            
            # 存储该用户的所有解释结果
            all_explanations[user_id_str] = user_explanations
            
            # 每处理10个用户就保存一次中间结果，防止程序中断丢失数据
            if (user_idx + 1) % 100 == 0:
                save_all_explanations_to_file(all_explanations, f"{DATASET}_{MODEL}_all_counterfactual_explanations_partial.json")
        
        # 保存所有解释结果到文件
        save_all_explanations_to_file(all_explanations)
        
        print("\n所有反事实解释生成完成！")
        
    except Exception as e:
        print(f"生成反事实解释时出错: {e}")
        import traceback
        traceback.print_exc()

def explain_for_specific_user_item(user_id_str, target_item_str, k=5):
    """
    为指定的用户和物品生成反事实解释
    
    Args:
        user_id_str: 用户ID
        target_item_str: 目标物品ID
        k: 返回前k个最重要的解释
    """

    # 初始化RecUtils
    rec_utils = RecUtils(model=MODEL, dataset=DATASET, config_file_list=config_file_list, config_dict=config)
    
    # 查找推荐结果文件
    rec_file = f"{DATASET}_{MODEL}_recommendations_top{topK}.json"
    # 检查文件是否存在，如果不存在尝试其他可能的文件
    if not os.path.exists(rec_file):
        possible_files = [f for f in os.listdir(".") if f.endswith("_recommendations_top50.json")]
        if possible_files:
            rec_file = possible_files[0]
    
    if not os.path.exists(rec_file):
        raise FileNotFoundError(f"未找到推荐结果文件: {rec_file}")
    
    # 查找模型文件
    model_file = find_model_file(rec_utils, MODEL, DATASET)
    
    if model_file is None or not os.path.exists(model_file):
        raise FileNotFoundError(f"未找到匹配的模型文件: {model_file}")
    
    print(f"=== 基于嵌入的反事实解释生成 ===")
    print(f"用户: {user_id_str}")
    print(f"推荐物品: {target_item_str}")
    print("=" * 50)
    
    try:
        # 创建基于嵌入的反事实解释器
        explainer = EmbeddingBasedCounterfactualExplainer(rec_utils, model_file)
        
        # 生成反事实解释
        explanations = explainer.explain(user_id_str, target_item_str, k=k)
        
        # 输出解释结果
        print("\n反事实解释结果:")
        print("=" * 50)
        for explanation in explanations:
            print(f"\n{explanation}")
            print("-" * 30)
        
        # 保存解释结果到文件
        # save_explanations_to_file(explanations, user_id_str, target_item_str)
        
        return explanations
    except Exception as e:
        print(f"生成反事实解释时出错: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    main()

    # explain_for_specific_user_item("1", "121", k=5)