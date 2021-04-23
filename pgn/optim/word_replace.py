from gensim.models import KeyedVectors, TfidfModel
from gensim.corpora import Dictionary
from data_utils import read_samples, isChinese, write_samples
from gensim import matutils
from gensim.models.word2vec import Word2Vec
from itertools import islice
import numpy as np

import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)


class EmbedReplace():
    def __init__(self, sample_path, wv_path):
        self.samples = read_samples(sample_path)
        self.refs = [sample.split('<SEP>')[1].split(' ') for sample in self.samples]
        self.wv = Word2Vec.load(wv_path)

        self.tfidf_model = TfidfModel.load(root_path + '/tf_idf/text_summary_baseline-5.tfidf')
        self.dict = Dictionary.load(root_path + '/tf_idf/text_summary_baseline-5.dict')
        self.corpus = [self.dict.doc2bow(doc) for doc in self.refs]
        self.vocab_size = len(self.dict.token2id)

    def extract_keywords(self, dic, tfidf, threshold=0.2, topk=5):
        tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)
        return list(islice([dic[w] for w, score in tfidf if score > threshold], topk))

    def replace(self, token_list, doc):
        keywords = self.extract_keywords(self.dict, self.tfidf_model[doc])
        num = int(len(token_list) * 0.3)
        new_tokens = token_list.copy()
        while num == int(len(token_list) * 0.3):
            indexes = np.random.choice(len(token_list), num)
            for index in indexes:
                token = token_list[index]
                if isChinese(token) and token not in keywords and token in self.wv:
                    new_tokens[index] = self.wv.most_similar(positive=token, negative=None, topn=1)[0][0]

            num -= 1

        return ' '.join(new_tokens)

    def generate_samples(self, write_path):
        replaced = []
        count = 0
        for sample, token_list, doc in zip(self.samples, self.refs, self.corpus):
            count += 1
            if count % 2000 == 0:
                print('count=', count)
                write_samples(replaced, write_path, 'a')
                replaced = []
            replaced.append(sample.split('<SEP>')[0] + '<SEP>' + self.replace(token_list, doc))