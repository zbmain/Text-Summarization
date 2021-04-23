# 导入相关工具包
import sys
import os
from collections import Counter
import torch
from torch.utils.data import Dataset

# 设置项目的root路径, 方便后续相关代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入项目中的自定义代码文件
from utils.func_utils import simple_tokenizer, count_words, sort_batch_by_len, source2ids, abstract2ids
from utils.vocab import Vocab
from utils import config


# 创建数据对的类
class PairDataset(object):
    def __init__(self, filename, tokenize=simple_tokenizer, max_enc_len=None, max_dec_len=None,
                 truncate_enc=False, truncate_dec=False):
        print("Reading dataset %s..." % filename, end=' ', flush=True)
        self.filename = filename
        self.pairs = []

        # 直接读取训练集数据文件, 切分成编码器数据, 解码器数据, 并做长度的统一
        with open(filename, 'r', encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                # 在数据预处理阶段已经约定好了x, y之间以<SEP>分隔
                pair = line.strip().split('<SEP>')
                if len(pair) != 2:
                    print("Line %d of %s is error formed." % (i, filename))
                    print(line)
                    continue
                # 前半部分是编码器数据, 即article原始文本.
                enc = tokenize(pair[0])
                if max_enc_len and len(enc) > max_enc_len:
                    if truncate_enc:
                        enc = enc[:max_enc_len]
                    else:
                        continue
                # 后半部分是解码器数据, 即abstract摘要文本
                dec = tokenize(pair[1])
                if max_dec_len and len(dec) > max_dec_len:
                    if truncate_dec:
                        dec = dec[:max_dec_len]
                    else:
                        continue
                # 以元组数据对的格式存储进结果列表中
                self.pairs.append((enc, dec))
        print("%d pairs." % len(self.pairs))

    # 构建模型所需的字典
    def build_vocab(self, embed_file=None):
        # 对读取的文件进行单词计数统计
        word_counts = Counter()
        count_words(word_counts, [enc + dec for enc, dec in self.pairs])
        # 初始化字典类
        vocab = Vocab()
        # 如果有预训练词向量就直接加载, 如果没有则随着模型一起训练获取
        vocab.load_embeddings(embed_file)

        # 将计数得到的结果写入字典类中
        for word, count in word_counts.most_common(config.max_vocab_size):
            vocab.add_words([word])

        # 返回在vocab.py代码文件中定义的字典类结果
        return vocab

# 直接为后续创建DataLoader提供服务的数据集预处理类
class SampleDataset(Dataset):
    def __init__(self, data_pair, vocab):
        self.src_sents = [x[0] for x in data_pair]
        self.trg_sents = [x[1] for x in data_pair]
        self.vocab = vocab
        self._len = len(data_pair)

    # 需要自定义__getitem__()取元素的函数
    def __getitem__(self, index):
        # 调用工具函数获取输入x和oov
        x, oov = source2ids(self.src_sents[index], self.vocab)
        # 完全按照模型需求, 自定义返回格式, 共有6个字段, 每个字段"个性化定制"即可
        return {'x': [self.vocab.SOS] + x + [self.vocab.EOS],
                'OOV': oov,
                'len_OOV': len(oov),
                'y': [self.vocab.SOS] + abstract2ids(self.trg_sents[index], self.vocab, oov) + [self.vocab.EOS],
                'x_len': len(self.src_sents[index]),
                'y_len': len(self.trg_sents[index])
                }

    def __len__(self):
        return self._len


# 创建DataLoader时自定义的数据处理函数
def collate_fn(batch):
    # 按照最大长度的限制, 对张量进行填充0
    def padding(indice, max_length, pad_idx=0):
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    # 对一个批次中的数据, 按照x_len字段进行排序
    data_batch = sort_batch_by_len(batch)

    # 依次取得所需的字段, 作为构建DataLoader的返回数据, 本模型需要6个字段
    x = data_batch['x']
    x_max_length = max([len(t) for t in x])
    y = data_batch['y']
    y_max_length = max([len(t) for t in y])

    OOV = data_batch['OOV']
    len_OOV = torch.tensor(data_batch['len_OOV'])

    x_padded = padding(x, x_max_length)
    y_padded = padding(y, y_max_length)

    x_len = torch.tensor(data_batch['x_len'])
    y_len = torch.tensor(data_batch['y_len'])

    return x_padded, y_padded, x_len, y_len, OOV, len_OOV