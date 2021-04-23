import requests
import time

# 定义请求url和传入的data
url = "http://0.0.0.0:5000/v1/main_server/"
data = {"uid": "AI-6-202104", "text": "丰田花冠行驶十万公里皮带要换正时皮带，这种车型10万公里最好更换一次。皮>带时间长会出现老化出现断裂，会损坏发动机，造成车辆抛锚。发电机皮带都需要检查一下。"}

start_time = time.time()
# 向服务发送post请求
res = requests.post(url, data=data)

cost_time = time.time() - start_time

# 打印返回的结果
print('文本摘要是: ', res.text)
print('单条样本耗时: ', cost_time * 1000, 'ms')

# python test.py