import random
import os
import sys
import torch
import jieba

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils import config
from src.model import PGN
from utils.dataset import PairDataset
from utils.func_utils import source2ids, outputids2words, Beam, timer, add2heap, replace_oovs


class Predict():
    @timer(module='initalize predicter')
    def __init__(self):
        # 此处的设备设定的是CPU
        self.DEVICE = config.DEVICE

        dataset = PairDataset(config.train_data_path,
                              max_enc_len=config.max_enc_len,
                              max_dec_len=config.max_dec_len,
                              truncate_enc=config.truncate_enc,
                              truncate_dec=config.truncate_dec)

        self.vocab = dataset.build_vocab(embed_file=config.embed_file)

        self.stop_word = list(set([self.vocab[x.strip()] for x in open(config.stop_word_file).readlines()]))


        # -------------------------------------------------------------------------------------------
        # 关于模型的量化在CPU上推断, Predict类中只有初始化函数__init__()中的如下部分需要修改.
        model = PGN(self.vocab)

        # 将在GPU上训练好的模型执行量化
        self.model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

        # 并将量化后的模型放到CPU上执行.
        self.model.to(self.DEVICE)
        # -------------------------------------------------------------------------------------------