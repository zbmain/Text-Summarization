# 导入工具包
import os
import sys
# 设置项目的root路径, 方便后续相关代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入相关工具包
from collections import Counter
import numpy as np
import torch
import torch.nn as nn

# 如果采用预训练词向量的策略, 则导入相关配置和工具包
from utils.config import word_vector_model_path
from gensim.models import word2vec


# 词典类的创建
class Vocab(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.word2index = {}
        self.word2count = Counter()
        self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.index2word = self.reserved[:]
        # 如果预训练词向量, 则后续直接载入; 否则置为None即可
        self.embedding_matrix = None

    # 向类词典中增加单词
    def add_words(self, words):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)

        # 因为引入Counter()工具包, 直接执行update()更新即可.
        self.word2count.update(words)

    # 如果已经提前预训练的词向量, 则执行类内函数对embedding_matrix赋值
    def load_embeddings(self, word_vector_model_path):
        # 直接下载预训练词向量模型
        wv_model = word2vec.Word2Vec.load(word_vector_model_path)
        # 从模型中直接提取词嵌入矩阵
        self.embedding_matrix = wv_model.wv.vectors

    # 根据id值item读取字典中的单词
    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    # 获取字典的当前长度(等效于单词总数)
    def __len__(self):
        return len(self.index2word)

    # 获取字典的当前单词总数
    def size(self):
        return len(self.index2word)