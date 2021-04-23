# 导入工具包并设置项目的root路径, 方便后续相关代码文件的导入
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入项目配置信息
from utils.config import *
# 导入gensim相关工具包
from gensim.corpora import Dictionary
from gensim import models
from gensim.models import word2vec

# 指定训练集数据路径, 这里采用70000条完整训练集数据进行TF-IDF模型的训练.
data_path = root_path + '/data/train.txt'

word_list = []
n = 0
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        article, abstract = line.strip('\n').split('<SEP>')
        text = article + ' ' + abstract
        word_list.append(text.split(' '))
        n += 1
        if n % 10000 == 0:
            print('n=', n)

print('n=', n)
print('word_list=', len(word_list))
print('***********************************')

print('开始创建数字化字典......')
dictionary = Dictionary(word_list)
new_corpus = [dictionary.doc2bow(text) for text in word_list]

saved_path = root_path + '/tf_idf/'
print('开始训练TF-IDF模型......')
tfidf = models.TfidfModel(new_corpus)

tfidf.save(saved_path + 'text_summary_baseline-5.tfidf')
print('保存TF-IDF模型完毕, 文件为text_summary_baseline-5.tfidf')

dictionary.save(saved_path + 'text_summary_baseline-5.dict')
print('保存TF-IDF字典完毕, 文件为text_summary_baseline-5.dict')