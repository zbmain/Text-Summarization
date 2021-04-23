# 导入gensim中的字典和模型工具包
from gensim.corpora import Dictionary
from gensim import models

# 模拟一段输入文本数据
corpus = ['this is the first document',
          'this is the second second document',
          'and the third one',
          'is this the first document'
         ]

# 手动分词并构造列表
word_list = []
for i in range(len(corpus)):
    word_list.append(corpus[i].split(' '))

# 将分词后的列表结构送入字典类中, 得到实例化对象.
dictionary = Dictionary(word_list)

# 调用.doc2bow()方法即可得到文本的字典映射.
# 结果元组中的第一个元素是单词在词典中对应的id, 第二个元素是单词在文档中出现的次数.
new_corpus = [dictionary.doc2bow(text) for text in word_list]

# 将单词到数字的映射字典打印出来.
print(dictionary.token2id)

# 将由字典Dictionary生成的数字化文档作为参数, 传入模型类中训练TF-IDF模型
tfidf = models.TfidfModel(new_corpus)

# 保存训练好的模型并打印
tfidf.save('demo.tfidf')
print(tfidf)