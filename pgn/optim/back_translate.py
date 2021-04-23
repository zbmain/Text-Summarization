import jieba
import time
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from optim.data_utils import write_samples

# 导入最新的Google翻译接口, 这个API接口已经和2020年夏天不同了, 未来还会发生变化.
from google_trans_new import google_translator
# 实例化翻译对象
translator = google_translator()

# 回译数据函数
def back_translate(batch_data):
    # 进行第一次翻译, 翻译目标是韩语
    ko_res = translator.translate(batch_data, lang_src='zh-cn', lang_tgt='ko')
    # 为了让调用延时顺利进行, 程序等待1秒钟.
    time.sleep(1.0)
    # 最后在翻译回中文, 完成回译全部流程
    cn_res = translator.translate(ko_res, lang_src='ko', lang_tgt='zh-cn')
    time.sleep(1.0)
    return cn_res

def translate(sample_path, translate_path):
    # 将原始文本sample_path经过回译数据法处理后, 写入到生成文本translate_path中.
    translated = []
    count = 0
    with open(sample_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # article调用back_translate, 并且abstract调用back_translate.
            article, abstract = line.strip('\n').split('<SEP>')
            article = ''.join(article.split(' '))
            abstract = ''.join(abstract.split(' '))

            # 传入回译函数的参数都是没有空格分隔的连续字符串
            back_article = back_translate(article)
            back_abstract = back_translate(abstract)

            # 回译结果需要重新经过分词, 然后以空格进行分隔, 以满足训练集数据的格式要求
            source = ' '.join(list(jieba.cut(back_article)))
            ref = ' '.join(list(jieba.cut(back_abstract)))
            # 文本和摘要之间还是以<SEP>进行分隔, 所有的数据格式保持一致.
            translated.append(source + '<SEP>' + ref)

            count += 1
            # 每处理20条数据后, 将回译数据后的样本存储进文件中.
            if count % 20 == 0:
                print('count=', count)
                # 每隔100条进行一次抽取打印, 以确保程序正常运行(尤其是Google接口正常服务).
                if count % 100 == 0:
                    print(translated[-1])
                # 回译后的数据追加模式写入文件中
                write_samples(translated, translate_path, 'a')
                # 将结果列表清空, 以为后续的数据添加做准备
                translated = []