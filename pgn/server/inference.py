import os
import sys
import torch
import time

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils import config
from utils.vocab import Vocab
from utils.dataset import PairDataset
from utils.func_utils import timer
from src.model import PGN

# --------------------------------------------------------------------------
# 代码文件inference.py中只有下面一行导入代码需要修改成量化的专门类Predict
from server.predict_quantized import Predict
# --------------------------------------------------------------------------

from server.rouge_eval import RougeEval


# 真实的测试机是val_data_path: dev.txt
print('实例化Rouge对象......')
rouge_eval = RougeEval(config.val_data_path)
print('实例化Predict对象......')
predict = Predict()

# 利用模型对article进行预测
print('利用模型对article进行预测, 并通过Rouge对象进行评估......')
rouge_eval.build_hypos(predict)

# 将预测结果和标签abstract进行ROUGE规则计算
print('开始用Rouge规则进行评估......')
result = rouge_eval.get_average()

print('rouge1: ', result['rouge-1'])
print('rouge2: ', result['rouge-2'])
print('rougeL: ', result['rouge-l'])


# 最后将计算评估结果写入文件中
print('将评估结果写入结果文件中......')
with open(root_path + '/eval_result/rouge_result_greedy_quantized_cpu.txt', 'w') as f:
    for r, metrics in result.items():
        f.write(r + '\n')
        for metric, value in metrics.items():
            f.write(metric + ': ' + str(value * 100) + '\n')