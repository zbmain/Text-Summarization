# 预处理文件
import os

# 设置项目的root目录, 方便后续代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 原始数据文本存储路径
train_raw_data_path = os.path.join(root_path, 'data', 'train.csv')
test_raw_data_path = os.path.join(root_path, 'data', 'test.csv')

# 停用词表和用户自定义字典的存储路径
stop_words_path = os.path.join(root_path, 'data', 'stopwords.txt')
user_dict_path = os.path.join(root_path, 'data', 'user_dict.txt')

# 预处理+切分后的训练测试数据路径
train_seg_path = os.path.join(root_path, 'data', 'train_seg_data.csv')
test_seg_path = os.path.join(root_path, 'data', 'test_seg_data.csv')

# 经过第一轮处理后的最终训练集数据和测试集数据
train_data_path = os.path.join(root_path, 'data', 'train.txt')
test_data_path = os.path.join(root_path, 'data', 'test.txt')

# 词向量模型路径
word_vector_path = os.path.join(root_path, 'data', 'wv', 'word2vec.model')