import json
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从通用配置文件导入配置参数
from config import MODEL, DATASET, TOPK as topK

def load_json_file(file_path):
    """Load JSON file and return its content"""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_forget_items(dataset_path):
    """
    从数据集文件加载用户的真实历史交互
    """
    items = []
    
    with open(dataset_path, 'r') as f:
        lines = f.readlines()[1:]  # 跳过标题行
        
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            item_id = parts[1]
            
            items.append(item_id)
    
    return items

def count_forget_items_in_top20(ori_rec, forget_items):
    """
    统计每个用户的top20推荐结果中有多少属于forget_items
    
    Args:
        ori_rec: 原始推荐结果字典
        forget_items: 需要遗忘的物品列表
    
    Returns:
        dict: 每个用户的统计结果
    """
    # 将forget_items转换为集合以提高查找效率
    forget_items_set = set(forget_items)
    
    # 存储每个用户的统计结果
    user_stats = {}
    
    # 遍历每个用户
    for user_id, recommendations in ori_rec.items():
        # 获取top20推荐物品
        # recommendations是一个字典，键是物品ID，值是评分
        # 按评分排序并取前20个
        top20_items = list(recommendations.keys())[:20]
        
        # 统计属于forget_items的数量
        count = 0
        forget_items_in_top20 = []
        for item_id in top20_items:
            if item_id in forget_items_set:
                count += 1
                forget_items_in_top20.append(item_id)
        
        # 保存统计结果
        user_stats[user_id] = {
            'count': count,
            'forget_items': forget_items_in_top20,
            'total_top20': len(top20_items)
        }
    
    return user_stats

def count_explanation_forget_intersections(explanations, forget_items):
    """
    遍历explanations和ori_rec，统计所有item的推荐列表的解释item列表和forgetset的交集
    
    Args:
        explanations: 反事实解释字典
        forget_items: 需要遗忘的物品列表
    
    Returns:
        dict: 每个用户的统计结果
    """
    # 将forget_items转换为集合以提高查找效率
    forget_items_set = set(forget_items)
    
    # 存储统计结果
    stats = {
        'user_stats': {},  # 每个用户的统计
    }
    
    # 遍历每个用户
    for user_id, user_explanations in explanations.items():
        user_stats = {
            'forget_items_in_explanations': [],  # 该用户解释中包含的遗忘物品
            'recommended_items_with_forget_explanations': 0  # 包含遗忘物品解释的推荐物品数
        }
        
        quota = 20
        # 遍历该用户每个推荐物品的解释
        for recommended_item, item_explanations in user_explanations.items():
            quota -= 1
            if quota <= 0:
                break
            # 检查解释中的物品是否在遗忘集中
            for explanation in item_explanations:
                explained_item = explanation['item_id']
                if explained_item in forget_items_set:
                    user_stats['forget_items_in_explanations'].append(explained_item)
                # break
        user_stats['recommended_items_with_forget_explanations'] = len(user_stats['forget_items_in_explanations'])
        stats['user_stats'][user_id] = user_stats
    
    return stats

# 加载原始推荐结果
ori_rec = load_json_file(f'{DATASET}_{MODEL}_recommendations_top50.json')

# print(ori_rec['1'])

# 加载遗忘集
forget_dataset_file = f'dataset/{DATASET}-forget/{DATASET}-forget.inter'  # 遗忘集
forget_items = load_forget_items(forget_dataset_file)
print(f"遗忘集物品数: {len(forget_items)}")

def load_explanations(file_path=f"{DATASET}_{MODEL}_all_counterfactual_explanations.json"):
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

# 加载解释
explanations = load_explanations()
print(f"加载了{len(explanations)}个用户的解释")

# 统计解释中包含的遗忘物品
explanation_forget_stats = count_explanation_forget_intersections(explanations, forget_items)


result = {}
for user_id, rec_list in ori_rec.items():
    user_stats = {
            'target_items': [], 
            'untarget_items': [] 
        }
    all_explain_items = explanation_forget_stats['user_stats'][user_id]['forget_items_in_explanations']
    all_explain_item_set = set(all_explain_items)
    
    target_items = []
    untarget_items = []
    for item_id, score in rec_list.items():
        # print(f"用户{user_id}的推荐列表中物品{item_id}的评分为{score}")
        if item_id in all_explain_item_set:
            print(f"用户{user_id}的推荐列表中包含遗忘物品{item_id}")

            target_items.append(item_id)
        else:
            untarget_items.append(item_id)
    user_stats['target_items'] = target_items
    user_stats['untarget_items'] = untarget_items
    result[user_id] = user_stats


output_file = f'{DATASET}_{MODEL}_explanation_forget_intersection_stats.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n详细统计结果已保存到 {output_file}")
