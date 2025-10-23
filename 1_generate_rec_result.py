import os
from recbole_utils import RecUtils
import pandas as pd
import os
import json
from tqdm import tqdm
from recbole.utils import set_color
from enum import Enum
def find_model_files(directory_path, model_name):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory_path):
        # 检查文件名是否包含 "abc"
        if model_name in filename and DATASET in filename:
            return os.path.join(directory_path, filename)

    return None

topK = 50
MODEL = "BPR"
# 处理的数据集
DATASET = "ml-100k"
# 默认配置文件， 注意 normalize_all: False 便于保留原始的时间和rating
config_files = f"config_file/{DATASET}.yaml"
config = {"normalize_all": False}
config_file_list = (
    config_files.strip().split(" ") if config_files else None
)


rec_utils = RecUtils(model=MODEL, dataset=DATASET, config_file_list=config_file_list, config_dict=config)

MODEL_FILE = find_model_files(directory_path=rec_utils.config["checkpoint_dir"], model_name=MODEL)
# 训练传统模型， 获得模型文件， 用于生成prompt的候选集
# MODEL_FILE = None
if MODEL_FILE is None:
    MODEL_FILE = rec_utils.train()

inter_path = os.path.join(rec_utils.config["data_path"], f"{DATASET}.inter")
# 加载数据
inter_df = pd.read_csv(inter_path, delimiter='\t')
# 获取唯一的用户ID并排序
unique_user_ids = sorted(inter_df["user_id:token"].unique())

# 创建一个字典来存储所有用户的推荐结果
recommendations = {}

for user_id in tqdm(unique_user_ids,
                    desc=set_color(f"Generating recommendations result (top-{topK}) ", "pink"),
                    unit="user"):
    # 获取推荐物品ID和对应的分数
    rec_with_scores = rec_utils.get_recommandation_list_with_scores(ori_user_id=user_id, topk=topK, model_file=MODEL_FILE)
    # 保存为 {item_id: score} 的字典格式
    recommendations[str(user_id)] = {item_id: float(score) for item_id, score in rec_with_scores}


# 将推荐结果保存到文件中
output_file = f"{DATASET}_{MODEL}_recommendations_top{topK}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(recommendations, f, ensure_ascii=False, indent=2)

print(f"推荐结果已保存到 {output_file}")