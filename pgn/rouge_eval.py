import os
import sys
from rouge import Rouge
import jieba

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from src.predict import Predict
from utils.func_utils import timer
from utils import config


class RougeEval():
    def __init__(self, path):
        self.path = path
        self.scores = None
        self.rouge = Rouge()
        self.sources = []
        self.hypos = []
        self.refs = []
        self.process()

    def process(self):
        print('Reading from ', self.path)
        with open(self.path, 'r') as test:
            for line in test:
                source, ref = line.strip().split('<SEP>')
                ref = ref.replace('。', '.')
                self.sources.append(source)
                self.refs.append(ref)

        print('self.refs[]包含的样本数: ', len(self.refs))
        print(f'Test set contains {len(self.sources)} samples.')

    @timer('building hypotheses')
    def build_hypos(self, predict):
        # Generate hypos for the dataset.
        print('Building hypotheses.')
        count = 0
        for source in self.sources:
            count += 1
            if count % 1000 == 0:
                print('count=', count)
            self.hypos.append(predict.predict(source.split()))

    def get_average(self):
        assert len(self.hypos) > 0, 'Build hypotheses first!'
        print('Calculating average rouge scores.')
        return self.rouge.get_scores(self.hypos, self.refs, avg=True)

    def one_sample(self, hypo, ref):
        return self.rouge.get_scores(hypo, ref)[0]