# 导入必备工具包
import os
import sys
import torch
import time
import jieba

# 设定项目的root路径, 方便后续相关代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 服务框架使用Flask, 导入工具包
from flask import Flask
from flask import request
app = Flask(__name__)

# 导入发送http请求的requests工具
import requests

# 导入项目相关的代码文件
from utils import config
from src.model import PGN
from utils.dataset import PairDataset
from utils.func_utils import source2ids, outputids2words, Beam, timer, add2heap, replace_oovs
# 导入专门针对CPU的预测类Predict
from server.predict_cpu import Predict

# 加载自定义的停用词字典
jieba.load_userdict(config.stop_word_file)

# 实例化Predict对象, 用于推断摘要, 提供服务请求
predict = Predict()
print('预测类Predict实例化完毕...')

# 设定文本摘要服务的路由和请求方法
@app.route('/v1/main_server/', methods=["POST"])
def main_server():
    # 接收来自请求方发送的服务字段
    uid = request.form['uid']
    text = request.form['text']

    # 对请求文本进行处理
    article = jieba.lcut(text)

    # 调用预测类对象执行摘要提取
    res = predict.predict(article)

    return res