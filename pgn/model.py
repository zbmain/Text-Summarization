# 导入系统工具包
import os
import sys
# 设置项目的root路径, 方便后续相关代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入若干工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入项目中的相关代码文件
from utils import config
from utils.func_utils import timer, replace_oovs
from utils.vocab import Vocab


# 构建编码器类
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_drop=0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, dropout=rnn_drop, batch_first=True)

    def forward(self, x, decoder_embedding):
        # ---------------------------------------------------------------------------
        if config.weight_tying:
            embedded = decoder_embedding(x)
        else:
            embedded = self.embedding(x)
        output, hidden = self.lstm(embedded)
        # ---------------------------------------------------------------------------

        return output, hidden

# 构建注意力类
class Attention(nn.Module):
    def __init__(self, hidden_units):
        super(Attention, self).__init__()
        # 定义前向传播层, 对应论文中的公式1中的Wh, Ws
        self.Wh = nn.Linear(2 * hidden_units, 2 * hidden_units, bias=False)
        self.Ws = nn.Linear(2 * hidden_units, 2 * hidden_units)


        # ---------------------------------------------------------------
        # 下面一行代码是baseline-3模型增加coverage机制的新增代码
        # 定义全连接层wc, 对应论文中的coverage处理
        self.wc = nn.Linear(1, 2 * hidden_units, bias=False)
        # ---------------------------------------------------------------


        # 定义全连接层, 对应论文中的公式1中最外层的v
        self.v = nn.Linear(2 * hidden_units, 1, bias=False)

    # 相比于baseline-2模型, 此处forward函数新增最后一个参数coverage_vector
    def forward(self, decoder_states, encoder_output, x_padding_masks, coverage_vector):
        h_dec, c_dec = decoder_states
        # 将两个张量在最后一个维度拼接, 得到deocder state St: (1, batch_size, 2*hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim=2)
        # 将batch_size置于第一个维度上: (batch_size, 1, 2*hidden_units)
        s_t = s_t.transpose(0, 1)
        # 按照hi的维度扩展St的维度: (batch_size, seq_length, 2*hidden_units)
        s_t = s_t.expand_as(encoder_output).contiguous()

         # 根据论文中的公式1来计算et, 总共有三步
        # 第一步: 分别经历各自的全连接层矩阵乘法
        # Wh * h_i: (batch_size, seq_length, 2*hidden_units)
        encoder_features = self.Wh(encoder_output.contiguous())
        # Ws * s_t: (batch_size, seq_length, 2*hidden_units)
        decoder_features = self.Ws(s_t)

        # 第二步: 两部分执行加和运算
        # (batch_size, seq_length, 2*hidden_units)
        attn_inputs = encoder_features + decoder_features


        # -----------------------------------------------------------------
        # 下面新增的3行代码是baseline-3为服务于coverage机制而新增的.
        if config.coverage:
            coverage_features = self.wc(coverage_vector.unsqueeze(2))
            attn_inputs = attn_inputs + coverage_features
        # -----------------------------------------------------------------



        # 第三步: 执行tanh运算和一个全连接层的运算
        # (batch_size, seq_length, 1)
        score = self.v(torch.tanh(attn_inputs))

        # 得到score后, 执行论文中的公式2
        # (batch_size, seq_length)
        attention_weights = F.softmax(score, dim=1).squeeze(2)

        # 添加一步执行padding mask的运算, 将编码器端无效的PAD字符全部遮掩掉
        attention_weights = attention_weights * x_padding_masks

        # 最整个注意力层执行一次正则化操作
        normalization_factor = attention_weights.sum(1, keepdim=True)
        attention_weights = attention_weights / normalization_factor

        # 执行论文中的公式3,将上一步得到的attention distributon应用在encoder hidden states上,得到context_vector
        # (batch_size, 1, 2*hidden_units)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_output)
        # (batch_size, 2*hidden_units)
        context_vector = context_vector.squeeze(1)


        # ----------------------------------------------------------------
        # 下面新增的2行代码是baseline-3模型为服务于coverage机制而新增的.
        # 按照论文中的公式10更新coverage vector
        if config.coverage:
            coverage_vector = coverage_vector + attention_weights
        # ----------------------------------------------------------------


        # 在baseline-2中我们返回2个张量; 在baseline-3中我们新增返回coverage vector张量.
        return context_vector, attention_weights, coverage_vector

# 构建解码器类
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, enc_hidden_size=None):
        super(Decoder, self).__init__()
        # 解码器端也采用跟随模型一起训练的方式, 得到词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # 解码器的主体结构采用单向LSTM, 区别于编码器端的双向LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        # 因为要将decoder hidden state和context vector进行拼接, 因此需要3倍的hidden_size维度设置
        self.W1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, vocab_size)

        if config.pointer:
            # 因为要根据论文中的公式8进行运算, 所谓输入维度上匹配的是4 * hidden_size + embed_size
            self.w_gen = nn.Linear(self.hidden_size * 4 + embed_size, 1)

    def forward(self, x_t, decoder_states, context_vector):
        # 首先计算Decoder的前向传播输出张量
        decoder_emb = self.embedding(x_t)
        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)

        # 接下来就是论文中的公式4的计算.
        # 将context vector和decoder state进行拼接, (batch_size, 3*hidden_units)
        decoder_output = decoder_output.view(-1, config.hidden_size)
        concat_vector = torch.cat([decoder_output, context_vector], dim=-1)

        # 经历两个全连接层V和V1后,再进行softmax运算, 得到vocabulary distribution
        # (batch_size, hidden_units)
        FF1_out = self.W1(concat_vector)


        # ---------------------------------------------------------------------------
        if config.weight_tying:
            FF2_out = torch.mm(FF1_out, torch.t(self.embedding.weight))
        else:
            FF2_out = self.W2(FF1_out)
        # (batch_size, vocab_size)
        p_vocab = F.softmax(FF2_out, dim=1)
        # ---------------------------------------------------------------------------


        # 构造decoder state s_t.
        h_dec, c_dec = decoder_states
        # (1, batch_size, 2*hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim=2)

        # p_gen是通过context vector h_t, decoder state s_t, decoder input x_t, 三个部分共同计算出来的.
        # 下面的部分是计算论文中的公式8.
        p_gen = None
        if config.pointer:
            # 这里面采用了直接拼接3部分输入张量, 然后经历一个共同的全连接层w_gen, 和原始论文的计算不同.
            # 这也给了大家提示, 可以提高模型的复杂度, 完全模拟原始论文中的3个全连接层来实现代码.
            x_gen = torch.cat([context_vector, s_t.squeeze(0), decoder_emb.squeeze(1)], dim=-1)
            p_gen = torch.sigmoid(self.w_gen(x_gen))

        return p_vocab, decoder_states, p_gen


# 构造加和state的类, 方便模型运算
class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim=True)
        c_reduced = torch.sum(c, dim=0, keepdim=True)
        return (h_reduced, c_reduced)

# 构建PGN类
class PGN(nn.Module):
    def __init__(self, v):
        super(PGN, self).__init__()
        # 初始化字典对象
        self.v = v
        self.DEVICE = config.DEVICE

        # 依次初始化4个类对象
        self.attention = Attention(config.hidden_size)
        self.encoder = Encoder(len(v), config.embed_size, config.hidden_size)
        self.decoder = Decoder(len(v), config.embed_size, config.hidden_size)
        self.reduce_state = ReduceState()

    # 计算最终分布的函数
    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights, max_oov):
        if not config.pointer:
            return p_vocab

        batch_size = x.size()[0]
        # 进行p_gen概率值的裁剪, 具体取值范围可以调参
        p_gen = torch.clamp(p_gen, 0.001, 0.999)
        # 接下来两行代码是论文中公式9的计算.
        p_vocab_weighted = p_gen * p_vocab
        # (batch_size, seq_len)
        attention_weighted = (1 - p_gen) * attention_weights

        # 得到扩展后的单词概率分布(extended-vocab probability distribution)
        # extended_size = len(self.v) + max_oovs
        extension = torch.zeros((batch_size, max_oov)).float().to(self.DEVICE)
        # (batch_size, extended_vocab_size)
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)

        # 根据论文中的公式9, 累加注意力值attention_weighted到对应的单词位置x
        final_distribution = p_vocab_extended.scatter_add_(dim=1, index=x, src=attention_weighted)

        return final_distribution

    def forward(self, x, x_len, y, len_oovs, batch, num_batches, teacher_forcing):
        x_copy = replace_oovs(x, self.v)
        x_padding_masks = torch.ne(x, 0).byte().float()


        # --------------------------------------------------------------------------------
        # 下面一行代码修改, 加入self.decoder.embedding是整个PGN类中唯一一行需要修改的代码.
        # 第一步: 进行Encoder的编码计算
        encoder_output, encoder_states = self.encoder(x_copy, self.decoder.embedding)
        # --------------------------------------------------------------------------------


        decoder_states = self.reduce_state(encoder_states)

        # 用全零张量初始化coverage vector.
        coverage_vector = torch.zeros(x.size()).to(self.DEVICE)

        # 初始化每一步的损失值
        step_losses = []

        # 第二步: 循环解码, 每一个时间步都经历注意力的计算, 解码器层的计算.
        # 初始化解码器的输入, 是ground truth中的第一列, 即真实摘要的第一个字符
        x_t = y[:, 0]
        for t in range(y.shape[1] - 1):
            # 如果使用Teacher_forcing, 则每一个时间步用真实标签来指导训练
            if teacher_forcing:
                x_t = y[:, t]

            x_t = replace_oovs(x_t, self.v)
            y_t = y[:, t + 1]

            # 通过注意力层计算context vector.
            context_vector, attention_weights, coverage_vector = self.attention(decoder_states,
                                                                                encoder_output,
                                                                                x_padding_masks,
                                                                                coverage_vector)

            # 通过解码器层计算得到vocab distribution和hidden states
            p_vocab, decoder_states, p_gen = self.decoder(x_t.unsqueeze(1), decoder_states, context_vector)

            # 得到最终的概率分布
            final_dist = self.get_final_distribution(x,p_gen,p_vocab,attention_weights,torch.max(len_oovs))

            # 第t个时间步的预测结果, 将作为第t + 1个时间步的输入(如果采用Teacher-forcing则不同).
            x_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)

            # 根据模型对target tokens的预测, 来获取到预测的概率
            if not config.pointer:
                y_t = replace_oovs(y_t, self.v)
            target_probs = torch.gather(final_dist, 1, y_t.unsqueeze(1))
            target_probs = target_probs.squeeze(1)

            # 将解码器端的PAD用padding mask遮掩掉, 防止计算loss时的干扰
            mask = torch.ne(y_t, 0).byte()
            # 为防止计算log(0)而做的数学上的平滑处理
            loss = -torch.log(target_probs + config.eps)

            # 关于coverage loss的处理逻辑代码.
            if config.coverage:
                # 按照论文中的公式12, 计算covloss.
                ct_min = torch.min(attention_weights, coverage_vector)
                cov_loss = torch.sum(ct_min, dim=1)
                # 按照论文中的公式13, 计算加入coverage机制后整个模型的损失值.
                loss = loss + config.LAMBDA * cov_loss


            # 先遮掩, 再添加损失值
            mask = mask.float()
            loss = loss * mask
            step_losses.append(loss)

        # 第三步: 计算一个批次样本的损失值, 为反向传播做准备.
        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)
        # 统计非PAD的字符个数, 作为当前批次序列的有效长度
        seq_len_mask = torch.ne(y, 0).byte().float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)

        # 计算批次样本的平均损失值
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss


if __name__ == '__main__':
    v = Vocab()
    model = PGN(v)
    print(model)