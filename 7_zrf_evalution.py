import json
import numpy as np
from scipy.spatial.distance import jensenshannon
import torch
from recbole_utils import RecUtils

def load_recommendations(file_path):
    """加载推荐结果文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_forget_set_users(forget_set_file):
    """加载forget set中的用户ID"""
    forget_users = set()
    with open(forget_set_file, 'r') as f:
        # 跳过标题行
        next(f)
        # 读取所有用户ID
        for line in f:
            if line.strip():  # 确保行不为空
                user_id = line.split('\t')[0]  # 用户ID在第一列
                # 确保user_id是有效的数字
                if user_id.isdigit():
                    forget_users.add(user_id)
    return forget_users

def softmax(logits, temperature=1.0):
    """对logits应用softmax函数"""
    logits = np.array(logits)
    logits = logits / temperature
    exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性
    return exp_logits / np.sum(exp_logits)

def compute_zrf_score(unlearned_file, forget_set_file):
    """
    计算ZRF分数，即遗忘模型和遗忘教师在被遗忘样本集Ef上的输出分布差异
    遗忘教师为未训练的LightGCN模型（随机初始化）
    
    参数:
    unlearned_file: 遗忘后模型的推荐结果文件路径
    forget_set_file: forget set文件路径
    
    返回:
    ZRF分数 (越接近1表示遗忘越完整)
    """
    # 加载遗忘后模型的推荐结果文件
    unlearned_results = load_recommendations(unlearned_file)
    
    # 加载forget set中的用户ID
    forget_users = load_forget_set_users(forget_set_file)
    
    # 调试信息
    print(f"遗忘后模型用户数: {len(unlearned_results)}")
    print(f"Forget set用户数: {len(forget_users)}")
    
    # 只考虑同时存在于遗忘后模型和forget set中的用户
    valid_forget_users = []
    for user_id in forget_users:
        if user_id in unlearned_results:
            valid_forget_users.append(user_id)
    
    print(f"同时存在于遗忘后模型和forget set中的用户数: {len(valid_forget_users)}")
    
    if not valid_forget_users:
        # 打印一些调试信息
        print("Forget set中的部分用户ID:", list(forget_users)[:10])
        print("遗忘后模型中的部分用户ID:", list(unlearned_results.keys())[:10])
        raise ValueError("没有同时存在于forget set和遗忘后模型中的用户")
    
    # 初始化未训练的LightGCN模型作为遗忘教师
    print("初始化未训练的LightGCN模型作为遗忘教师...")
    config = {"normalize_all": False}
    config_file_list = ["config_file/ml-100k.yaml"]
    rec_utils = RecUtils(model="LightGCN", dataset="ml-100k", config_file_list=config_file_list, config_dict=config)
    
    # 计算每个用户的Jensen-Shannon散度
    js_divergences = []
    
    for user_id in valid_forget_users:
        # 获取遗忘后模型对该用户的推荐项目和分数
        unlearned_items_scores = unlearned_results[user_id]
        
        # 提取项目ID
        items = list(unlearned_items_scores.keys())
        
        # 使用未训练的LightGCN模型生成教师模型的分数
        try:
            # 将用户ID和项目ID转换为内部ID
            internal_user_id = rec_utils.get_encode_user_token(user_id)
            teacher_scores = []
            
            # 为每个项目获取教师模型的分数
            for item_id in items:
                internal_item_id = rec_utils.get_encode_item_token(item_id)
                # 使用未训练的模型获取分数
                rec_utils.model.eval()
                interaction = {"user_id": torch.tensor([internal_user_id]).to(rec_utils.config["device"])}
                # 创建一个简单的交互对象来获取分数
                with torch.no_grad():
                    # 获取用户嵌入
                    user_e = rec_utils.model.user_embedding(torch.tensor([internal_user_id]).to(rec_utils.config["device"]))
                    # 获取项目嵌入
                    item_e = rec_utils.model.item_embedding(torch.tensor([internal_item_id]).to(rec_utils.config["device"]))
                    # 计算点积作为分数
                    score = torch.mul(user_e, item_e).sum(dim=1).item()
                    teacher_scores.append(score)
            
            # 将遗忘后模型的分数转换为概率分布（使用softmax）
            unlearned_scores = [unlearned_items_scores[item] for item in items]
            unlearned_probs = softmax(unlearned_scores, temperature=1.0)
            
            # 将教师模型的分数转换为概率分布（使用softmax）
            teacher_probs = softmax(teacher_scores, temperature=1.0)
            
            # 计算Jensen-Shannon散度并归一化到[0,1]区间
            js_div = jensenshannon(unlearned_probs, teacher_probs)
            # 归一化到[0,1]区间 (JSD_norm = JSD / ln(2))
            js_div_normalized = js_div / np.log(2)
            js_divergences.append(js_div_normalized)
        except Exception as e:
            print(f"处理用户 {user_id} 时出错: {e}")
            # 如果无法获取教师模型分数，使用随机分数作为备选
            try:
                unlearned_scores = [unlearned_items_scores[item] for item in items]
                unlearned_probs = softmax(unlearned_scores, temperature=1.0)
                
                # 使用随机分数作为教师模型
                teacher_scores = np.random.rand(len(items))
                teacher_probs = softmax(teacher_scores, temperature=1.0)
                
                # 计算Jensen-Shannon散度并归一化到[0,1]区间
                js_div = jensenshannon(unlearned_probs, teacher_probs)
                js_div_normalized = js_div / np.log(2)
                js_divergences.append(js_div_normalized)
            except Exception as e2:
                print(f"处理用户 {user_id} 的备选方案也失败: {e2}")
                continue
    
    if not js_divergences:
        raise ValueError("没有足够的用户来计算Jensen-Shannon散度")
    
    # 计算平均Jensen-Shannon散度
    mean_js_div = np.mean(js_divergences)
    
    # ZRF分数定义为归一化后的Jensen-Shannon散度，越接近1表示遗忘越完整
    zrf_score = mean_js_div
    
    return zrf_score

def main():
    # 文件路径
    unlearned_file = 'ml-100k_LightGCN_recommendations_top20_updated.json'
    forget_set_file = 'dataset/ml-100k-forget/ml-100k-forget.inter'
    
    try:
        # 计算ZRF分数
        zrf_score = compute_zrf_score(unlearned_file, forget_set_file)
        print(f"ZRF Score: {zrf_score}")
        print(f"ZRF Score (作为遗忘完整性的度量，越接近1表示遗忘越完整): {zrf_score}")
    except Exception as e:
        print(f"计算ZRF分数时出错: {e}")

if __name__ == "__main__":
    main()