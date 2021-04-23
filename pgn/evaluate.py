# 导入工具包
import os
import sys
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

# 设定项目的root路径, 方便后续代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入项目的相关代码文件
from utils.dataset import collate_fn
from utils import config


# 编写评估函数
def evaluate(model, val_data, epoch):
    print('validating')
    val_loss = []
    # 评估模型需要设定参数不变
    with torch.no_grad():
        DEVICE = config.DEVICE
        # 创建数据迭代器, pin_memory=True是对于GPU机器的优化设置
        # 为了PGN模型数据的特殊性, 传入自定义的collate_fn提供个性化服务
        val_dataloader = DataLoader(dataset=val_data,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True,
                                    collate_fn=collate_fn)

        # 遍历测试集数据进行评估
        for batch, data in enumerate(tqdm(val_dataloader)):
            x, y, x_len, y_len, oov, len_oovs = data
            if config.is_cuda:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                x_len = x_len.to(DEVICE)
                len_oovs = len_oovs.to(DEVICE)
            total_num = len(val_dataloader)

            loss = model(x, x_len, y, len_oovs, batch=batch, num_batches=total_num, teacher_forcing=True)
            val_loss.append(loss.item())
    # 返回整个测试集的平均损失值
    return np.mean(val_loss)