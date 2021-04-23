import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)


def read_samples(filename):
    samples = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(line.strip('\n'))
        return samples


def write_samples(samples, file_path, opt='w'):
    with open(file_path, opt, encoding='utf-8') as f:
        for line in samples:
            f.write(line + '\n')


def isChinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True

    return False