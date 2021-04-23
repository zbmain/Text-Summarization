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


# 打印模型大小
def print_size_of_model(model):
    # 保存模型中的参数部分到持久化文件
    torch.save(model.state_dict(), "temp.p")
    # 打印持久化文件的大小
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    # 移除该文件
    os.remove('temp.p')


@timer(module='initalize and quantize model...')
def quantize_model(origin_model_path):
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
    print('-------------------------------------------------------------------')
    # 使用torch.quantization.quantize_dynamic获得动态量化的模型
    # 量化的网络层为所有的nn.Linear的权重，使其成为int8
    quantized_model_1 = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # 将在GPU上训练好的模型加载到GPU上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    quantized_model_1.to(device)

    # quantized_model_2 = torch.quantization.quantize_dynamic(model, 
    #                     {torch.nn.Linear, torch.nn.LSTM, torch.nn.Embedding}, dtype=torch.qint8)

    # 打印动态量化后的模型
    print(quantized_model)
    print('-------------------------------------------------------------------')

    # 分别打印model和quantized_model
    print_size_of_model(model)
    print_size_of_model(quantized_model)
    print('-------------------------------------------------------------------')

    # 将在GPU上训练好的模型加载到CPU上
    model.load_state_dict(torch.load(origin_model_path, map_location=lambda storage, loc:storage))
    print(model)
    model.to('cpu')