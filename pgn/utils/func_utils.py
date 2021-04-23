# 导入系统工具包
import os
import sys
# 设置项目的root路径, 方便后续相关代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入项目中的工具包
import numpy as np
import time
import heapq
import random
import pathlib
from utils import config
import torch


# 函数耗时计量函数
def timer(module):
    def wrapper(func):
        # func: 一个函数名, 下面的计时函数就是计算这个func函数的耗时.
        def cal_time( *args, **kwargs):
            t1 = time.time()
            res = func( *args, **kwargs)
            t2 = time.time()
            cost_time = t2 - t1
            print(f'{cost_time} secs used for ', module)
            return res
        return cal_time
    return wrapper

# 将一段文本按空格切分, 返回结果列表
def simple_tokenizer(text):
    return text.split()

# 以字典计数的方式统计一段文本中不同单词的数量
def count_words(counter, text):
    for sentence in text:
        for word in sentence:
            counter[word] += 1

# 对一个批次batch_size个样本, 按照x_len字段长短进行排序, 并返回排序后的结果
def sort_batch_by_len(data_batch):
    # 初始化一个结果字典, 其中包含的6个字段都是未来数据迭代器中的6个字段
    res = {'x': [],
           'y': [],
           'x_len': [],
           'y_len': [],
           'OOV': [],
           'len_OOV': []}

    # 遍历批次数据, 分别将6个字段数据按照字典key值, 添加进各自的列表中
    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append(data_batch[i]['len_OOV'])

    # 以x_len字段大小进行排序, 并返回下标结果的列表
    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()

    # 返回的data_batch依然保持字典类型
    data_batch = {name: [_tensor[i] for i in sorted_indices] for name, _tensor in res.items()}

    return data_batch

# 原始文本映射成ids张量
def source2ids(source_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.UNK
    for w in source_words:
        i = vocab[w]
        if i == unk_id:  # 如果w是OOV单词
            if w not in oovs:  # 将w添加进OOV列表中
                oovs.append(w)
            # 索引0对应第一个source document OOV, 索引1对应第二个source document OOV, 以此类推......
            oov_num = oovs.index(w)
            # 在本项目中索引vocab_size对应第一个source document OOV, vocab_size+1对应第二个source document OOV
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)
    return ids, oovs

# 摘要文本映射成数字化ids张量
def abstract2ids(abstract_words, vocab, source_oovs):
    ids = []
    unk_id = vocab.UNK
    for w in abstract_words:
        i = vocab[w]
        if i == unk_id:  # 如果w是OOV单词
            if w in source_oovs:  # 如果w是source document OOV
                # 对这样的w计算出一个新的映射id值
                vocab_idx = vocab.size() + source_oovs.index(w)
                ids.append(vocab_idx)
            else:  # 如果w不是一个source document OOV
                ids.append(unk_id)  # 对这样的w只能用UNK的id值来代替
        else:
            ids.append(i) # 如果w是词表中的单词, 直接取id值映射
    return ids

# 将输出张量ids结果映射成自然语言文本
def outputids2words(id_list, source_oovs, vocab):
    words = []
    for i in id_list:
        try:
            # w可能是<UNK>
            w = vocab.index2word[i]
        # w是OOV单词
        except IndexError:
            assert_msg = "Error: 无法在词典中找到该ID值."
            assert source_oovs is not None, assert_msg
            # 寄希望索引i是一个source document OOV单词
            source_oov_idx = i - vocab.size()
            try:
                # 如果成功取到, 则w是source document OOV单词
                w = source_oovs[source_oov_idx]
            # i不仅是OOV单词, 也不对应source document中的原文单词
            except ValueError:
                raise ValueError('Error: 模型生成的ID: %i, 原始文本中的OOV ID: %i \
                                  但是当前样本中只有%i个OOVs'
                                  % (i, source_oov_idx, len(source_oovs)))
        # 向结果列表中添加原始字符
        words.append(w)
    # 空格连接成字符串返回
    return ' '.join(words)

# 创建小顶堆, 包含k个点的特殊二叉树, 始终保持二叉树中最小的值在root根节点
def add2heap(heap, item, k):
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)

# 将文本张量中所有OOV单词的id, 全部替换成<UNK>对应的id
def replace_oovs(in_tensor, vocab):
    # oov_token = torch.full(in_tensor.shape, vocab.UNK).long().to(config.DEVICE)
    # 在Pytorch1.5.0以及更早的版本中, torch.full()默认返回float类型
    # 在Pytorch1.7.0最新版本中, torch.full()会将bool返回成torch.bool, 会将integer返回成torch.long.
    # 上面一行代码在Pytorch1.6.0版本中会报错, 因为必须指定成long类型, 如下面代码所示
    oov_token = torch.full(in_tensor.shape, vocab.UNK, dtype=torch.long).to(config.DEVICE)

    out_tensor = torch.where(in_tensor > len(vocab) - 1, oov_token, in_tensor)
    return out_tensor

# 获取模型训练中若干超参数信息
def config_info(config):
    info = 'model_name = {}, pointer = {}, coverage = {}, fine_tune = {}, scheduled_sampling = {}, weight_tying = {},' + 'source = {}  '
    return (info.format(config.model_name, config.pointer, config.coverage, config.fine_tune, config.scheduled_sampling, config.weight_tying, config.source))

# 构建Beam-search的基础类
class Beam(object):
    def __init__(self, tokens, log_probs, decoder_states, coverage_vector):
        # Beam类所需的4个参数
        # tokens: 已经搜索到的字符序列
        # log_probs: 已经搜索的字符序列的得分序列
        # decoder_states: Decoder解码器端的隐藏层状态张量
        # coverage_vector: 引入coverage机制后计算得到的coverage张量
        self.tokens = tokens
        self.log_probs = log_probs
        self.decoder_states = decoder_states
        self.coverage_vector = coverage_vector

    # 非常重要的扩展函数, 当前搜索序列向前进一步, 添加当前搜索的字符token和分数log_probs
    def extend(self, token, log_prob, decoder_states, coverage_vector):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    decoder_states=decoder_states,
                    coverage_vector=coverage_vector)

    # 计算当前序列得分的函数
    def seq_score(self):
        # 得到当前序列的长度, 用来计算正则化参数
        len_Y = len(self.tokens)
        # 序列长度的正则化, 自定义公式
        ln = (5 + len_Y)**config.alpha / (5 + 1)**config.alpha
        # coverage张量的正则化计算, 固定公式计算即可
        cn = config.beta * torch.sum(torch.log(config.eps + torch.where(self.coverage_vector < 1.0,
                                                                        self.coverage_vector,
                           torch.ones((1, self.coverage_vector.shape[1])).to(torch.device(config.DEVICE)))))

        # 直接利用上面的正则化参数, 计算当前序列的分数即可
        score = sum(self.log_probs) / ln + cn
        return score

    # 比较序列分数的小于<关系
    def __lt__(self, other):
        return self.seq_score() < other.seq_score()

    # 比较序列分数的小于等于<=关系
    def __le__(self, other):
        return self.seq_score() <= other.seq_score()

    # 本函数的作用是维护一个小顶堆, 拥有k个节点的二叉树结构, 最小值始终保持在堆顶!
    def add2heap(heap, item, k):
        # 如果当前堆的元素个数小于k, 则添加新节元素item为一个新的节点, 同时维护小顶堆的规则.
        if len(heap) < k:
            heapq.heappush(heap, item)
        # 如果当前堆的元素个数不小于k, 则添加新节元素item为一个新的节点, 同时按照小顶堆的规则删除一个不符合的节点.
        else:
            heapq.heappushpop(heap, item)

# 随着训练迭代步数的增长, 计算是否使用Teacher-forcing的概率大小
class ScheduledSampler():
    def __init__(self, phases):
        self.phases = phases
        # 通过超参数phases来提前计算出每一个epoch是否采用Teacher forcing的阈值概率
        self.scheduled_probs = [i / (self.phases - 1) for i in range(self.phases)]

    def teacher_forcing(self, phase):
        # 生成随机数
        sampling_prob = random.random()
        # 每一轮训练时, 通过随机数和阈值概率比较, 来决定是否采用Teacher forcing
        if sampling_prob >= self.scheduled_probs[phase]:
            return True
        else:
            return False