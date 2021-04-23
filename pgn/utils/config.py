import torch
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 神经网络通用参数
hidden_size = 512
dec_hidden_size = 512
embed_size = 512
pointer = True

# 模型相关配置参数
max_vocab_size = 20000
model_name = 'pgn_model'
embed_file = root_path + '/wv/word2vec_pad.model'
source = 'train'
train_data_path = root_path + '/data/train.txt'
val_data_path = root_path + '/data/dev.txt'
test_data_path = root_path + '/data/test.txt'
stop_word_file = root_path + '/data/stopwords.txt'
losses_path = root_path + '/data/loss.txt'
log_path = root_path + '/data/log_train.txt'
word_vector_model_path = root_path + '/wv/word2vec_pad.model'
encoder_save_name = root_path + '/saved_model/model_encoder.pt'
decoder_save_name = root_path + '/saved_model/model_decoder.pt'
attention_save_name = root_path + '/saved_model/model_attention.pt'
reduce_state_save_name = root_path + '/saved_model/model_reduce_state.pt'
model_save_path = root_path + '/saved_model/pgn_model.pt'
max_enc_len = 300
max_dec_len = 100
truncate_enc = True
truncate_dec = True
# 下面两个参数关系到predict阶段的展示效果, 需要按业务场景调参
min_dec_steps = 30
# 在Greedy Decode的时候设置为50
# max_dec_steps = 50
# 在Beam-search Decode的时候设置为30
max_dec_steps = 30
enc_rnn_dropout = 0.5
enc_attn = True
dec_attn = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0

# 训练参数
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.001
lr_decay = 0.0
initial_accumulator_value = 0.1
epochs = 10
batch_size = 32
is_cuda = True

# 下面4个参数都是第六章的优化策略
coverage = True
fine_tune = False
scheduled_sampling = True
weight_tying = True

max_grad_norm = 2.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA = 1

# Beam-search配置
beam_size = 3
alpha = 0.2
beta = 0.2
gamma = 2000