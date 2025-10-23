import sys
sys.path.append('/root/haichao/LLM_rec_unlearning')
from tqdm import tqdm
from recbole.utils import set_color
import pandas as pd
import torch
import json
from recbole_utils import RecUtils

# 读取 JSON 文件

file_path_list = ['ml-1m_LightGCN_recommendations_top50.json',
                  'ml-1m-remain_LightGCN_remainset_top50.json',
                  'ml-1m_LightGCN_recommendations_top20_updated.json']
# 获取candidate item 的传统推荐模型
MODEL = "LightGCN"
# 处理的数据集
DATASET = "ml-1m"
# 默认配置文件， 注意 normalize_all: False 便于保留原始的时间和rating
topK = [10,20]
config_files = f"config_file/{DATASET}.yaml"
config = {"normalize_all": False, "topk": topK}
config_file_list = (
    config_files.strip().split(" ") if config_files else None
)

rec_utils = RecUtils(model=MODEL, dataset=DATASET, config_file_list=config_file_list, config_dict = config)

def get_gpu_usage(device=None):
    r"""Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    return "{:.2f} G/{:.2f} G".format(reserved, total)


for file_path in file_path_list:
    # 读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化张量
    topk_idx = torch.zeros((rec_utils.dataset.user_num-1, max(topK)), dtype=torch.int64)

    exception_user = []
    
    # 使用 tqdm 显示进度条
    for user_id, user_recommendations in tqdm(
        data.items(),
        total=len(data),
        ncols=100,
        desc=set_color(f"Parse rec result  ", "pink"),
    ):
        try:
            encode_user_id = rec_utils.get_encode_user_token(user_id)
            
            # 处理不同的推荐结果格式
            sorted_items = sorted(user_recommendations.items(), key=lambda x: x[1], reverse=True)
            topK_ori_item_id = [item_id for item_id, score in sorted_items]
        
            # 如果推荐列表长度不足，用-1填充
            if len(topK_ori_item_id) < max(topK):
                topK_ori_item_id.extend([-1] * (max(topK) - len(topK_ori_item_id)))

            topK_encode_item_id = []
            for ori_item_id in topK_ori_item_id[:max(topK)]:
                encode_item_id = rec_utils.get_encode_item_token(ori_item_id)
                topK_encode_item_id.append(encode_item_id)
            topk_idx[encode_user_id - 1] = torch.tensor(topK_encode_item_id, dtype=torch.int64)

        except Exception as e:
            exception_user.append(user_id)

    pos_matrix = torch.zeros((rec_utils.dataset.user_num-1, rec_utils.dataset.item_num), dtype=torch.int64)

    iter_data = (
        tqdm(
            rec_utils.test_data,
            total=len(rec_utils.test_data),
            ncols=100,
            desc=set_color(f"Evaluate   ", "pink"),
        )
    )
    row_idx = 0
    for batch_idx, batched_data in enumerate(iter_data):
        interaction, _, positive_u, positive_i = batched_data
        pos_matrix[row_idx+positive_u, positive_i] = 1
        row_idx = row_idx + torch.unique(positive_u).numel()


    pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
    pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
    result = torch.cat((pos_idx, pos_len_list), dim=1)

    print(f"result.shape : {result.shape}")
    print(f"exception_user: {exception_user}")

    from recbole.evaluator.collector import DataStruct
    data_struct = DataStruct()
    data_struct.update_tensor("rec.topk", result)

    from recbole.evaluator.metrics import *

    print(f"{file_path} result: ")

    hit = Hit(rec_utils.config)
    metric_val = hit.calculate_metric(data_struct)
    print(metric_val)

    ndcg = NDCG(rec_utils.config)
    metric_val = ndcg.calculate_metric(data_struct)
    print(metric_val)

    print("\n")