import sys
import os
import torch
import time

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils import config
from utils.vocab import Vocab
from utils.dataset import PairDataset
from utils.func_utils import timer
from src.model import PGN


@timer(module='initalize and transform model...')
def transform_GPU_to_CPU(origin_model_path, to_device='cpu'):
    # 第一步: 构造数据集字典
    dataset = PairDataset(config.train_data_path,
                          max_enc_len=config.max_enc_len,
                          max_dec_len=config.max_dec_len,
                          truncate_enc=config.truncate_enc,
                          truncate_dec=config.truncate_dec)

    # 第二步: 利用字典, 实例化模型对象
    vocab = dataset.build_vocab(embed_file=config.embed_file)
    model = PGN(vocab)

    # 判断待加载的模型是否存在
    if not os.path.exists(origin_model_path):
        print('The model file is not exists!')
        exit(0)

    model.load_state_dict(torch.load(origin_model_path))
    print(model)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    print('-------------------------------------------------------------------')

    if to_device != 'cpu':
        print('Transform model to CPU!')
        exit(0)

    # 将在GPU上训练好的模型加载到CPU上
    model.load_state_dict(torch.load(origin_model_path, map_location=lambda storage, loc:storage))
    print(model)
    model.to(to_device)