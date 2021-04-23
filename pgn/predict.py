# 导入相关系统工具包
import random
import os
import sys
import torch
import jieba

# 设置项目的root路径, 方便后续相关代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入项目的相关代码文件
from utils import config
from src.model import PGN
from utils.dataset import PairDataset
from utils.func_utils import source2ids, outputids2words, Beam, timer, add2heap, replace_oovs


# 构建核心预测类Predict
class Predict():
    @timer(module='initalize predicter')
    def __init__(self):
        self.DEVICE = config.DEVICE

        # 产生数据对, 为接下来的迭代器做数据准备, 注意这里面用的是训练集数据
        dataset = PairDataset(config.train_data_path,
                              max_enc_len=config.max_enc_len,
                              max_dec_len=config.max_dec_len,
                              truncate_enc=config.truncate_enc,
                              truncate_dec=config.truncate_dec)

        # 生成词汇表
        self.vocab = dataset.build_vocab(embed_file=config.embed_file)

        # 实例化PGN模型类, 这里面的模型是基于baseline-3的模型.
        self.model = PGN(self.vocab)
        self.stop_word = list(set([self.vocab[x.strip()] for x in open(config.stop_word_file).readlines()]))

        # --------------------------------------------------------------------------------------
        # 下面两行代码是将模型加载到CPU上的核心部分, 也是相比于baseline-4模型唯一变动的地方.
        # 将在GPU上训练好的模型加载到CPU上
        self.model.load_state_dict(torch.load(config.model_save_path,
                                              map_location=lambda storage,
                                              loc:storage))

        self.model.to(self.DEVICE)
        # --------------------------------------------------------------------------------------


    def greedy_search(self, x, max_sum_len, len_oovs, x_padding_masks):
        encoder_output, encoder_states = self.model.encoder(replace_oovs(x, self.vocab), None)

        # 用encoder的hidden state初始化decoder的hidden state
        decoder_states = self.model.reduce_state(encoder_states)

        # 利用SOS作为解码器的初始化输入字符
        x_t = torch.ones(1) * self.vocab.SOS
        x_t = x_t.to(self.DEVICE, dtype=torch.int64)
        summary = [self.vocab.SOS]
        coverage_vector = torch.zeros((1, x.shape[1])).to(self.DEVICE)

        # 循环解码, 最多解码max_sum_len步
        while int(x_t.item()) != (self.vocab.EOS) and len(summary) < max_sum_len:
            context_vector, attention_weights = self.model.attention(decoder_states,
                                                                     encoder_output,
                                                                     x_padding_masks,
                                                                     coverage_vector)

            p_vocab, decoder_states, p_gen = self.model.decoder(x_t.unsqueeze(1),
                                                                decoder_states,
                                                                context_vector)

            final_dist = self.model.get_final_distribution(x, p_gen, p_vocab,
                                                           attention_weights,
                                                           torch.max(len_oovs))

            # 以贪心解码策略预测字符
            x_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)
            decoder_word_idx = x_t.item()

            # 将预测的字符添加进结果摘要中
            summary.append(decoder_word_idx)
            x_t = replace_oovs(x_t, self.vocab)

        return summary


    # 利用解码器产生出一个vocab distribution, 来预测下一个token.
    def best_k(self, beam, k, encoder_output, x_padding_masks, x, len_oovs):
        # beam: 代表Beam类的一个实例化对象.
        # k: 代表Beam-search中的重要参数beam_size=k.
        # encoder_output: 编码器的输出张量.
        # x_padding_masks: 输入序列的padding mask, 用于遮掩那些无效的PAD位置字符
        # x: 编码器的输入张量.
        # len_oovs: OOV列表的长度.
        # 当前时间步t的对应解析字符token, 将作为Decoder端的输入, 产生最终的vocab distribution.
        x_t = torch.tensor(beam.tokens[-1]).reshape(1, 1)
        x_t = x_t.to(self.DEVICE)

        # 通过注意力层attention, 得到context_vector
        context_vector, attention_weights, coverage_vector = self.model.attention(beam.decoder_states,
                                                                                  encoder_output,
                                                                                  x_padding_masks,
                                                                                  beam.coverage_vector)

        # 函数replace_oovs()将OOV单词替换成新的id值, 来避免解码器出现index-out-of-bound error
        p_vocab, decoder_states, p_gen = self.model.decoder(replace_oovs(x_t, self.vocab),
                                                            beam.decoder_states,
                                                            context_vector)

        # 调用PGN网络中的函数, 得到最终的单词分布(包含OOV)
        final_dist = self.model.get_final_distribution(x, p_gen, p_vocab,
                                                       attention_weights,
                                                       torch.max(len_oovs))

        # 计算序列的log_probs分数
        log_probs = torch.log(final_dist.squeeze())
        # 如果当前Beam序列只有1个token, 要将一些无效字符删除掉, 以免影响序列的计算.
        # 至于这个无效字符的列表都包含什么, 也是利用bad case的分析, 结合数据观察得到的, 属于调优的一部分.
        if len(beam.tokens) == 1:
            forbidden_ids = [self.vocab[u"这"],
                             self.vocab[u"此"],
                             self.vocab[u"采用"],
                             self.vocab[u"，"],
                             self.vocab[u"。"]]

            log_probs[forbidden_ids] = -float('inf')
        # 对于EOS token的一个罚分处理.
        # 具体做法参考了https://opennmt.net/OpenNMT/translation/beam_search/.
        log_probs[self.vocab.EOS] *= config.gamma * x.size()[1] / len(beam.tokens)
        log_probs[self.vocab.UNK] = -float('inf')

        # 从log_probs中获取top_k分数的tokens, 这也正好符合beam-search的逻辑.
        topk_probs, topk_idx = torch.topk(log_probs, k)

        # 非常关键的一行代码: 利用top_k的单词, 来扩展beam-search搜索序列, 等效于将top_k单词追加到候选序列的末尾.
        best_k = [beam.extend(x, log_probs[x], decoder_states, coverage_vector) for x in topk_idx.tolist()]

        # 返回追加后的结果列表
        return best_k


    # 支持beam-search解码策略的主逻辑函数.
    def beam_search(self, x, max_sum_len, beam_width, len_oovs, x_padding_masks):
        # x: 编码器的输入张量, 即article(source document)
        # max_sum_len: 本质上就是最大解码长度max_dec_len
        # beam_size: 采用beam-search策略下的搜索宽度k
        # len_oovs: OOV列表的长度
        # x_padding_masks: 针对编码器的掩码张量, 把无效的PAD字符遮掩掉.
        # 第一步: 通过Encoder计算得到编码器的输出张量.
        encoder_output, encoder_states = self.model.encoder(replace_oovs(x, self.vocab))

        # 全零张量初始化coverage vector
        coverage_vector = torch.zeros((1, x.shape[1])).to(self.DEVICE)
        # 对encoder_states进行加和降维处理, 赋值给decoder_states.
        decoder_states = self.model.reduce_state(encoder_states)

        # 初始化hypothesis, 第一个token给SOS, 分数给0.
        init_beam = Beam([self.vocab.SOS], [0], decoder_states, coverage_vector)

        # beam_size本质上就是搜索宽度k
        k = beam_size
        # 初始化curr作为当前候选集, completed作为最终的hypothesis列表
        curr, completed = [init_beam], []

        # 通过for循环连续解码max_sum_len步, 每一步应用beam-search策略产生预测token.
        for _ in range(max_sum_len):
            # 初始化当前时间步的topk列表为空, 后续将beam-search的解码结果存储在topk中.
            topk = []
            for beam in curr:
                # 如果产生了一个EOS token, 则将beam对象追加进最终的hypothesis列表, 并将k值减1, 然后继续搜索.
                if beam.tokens[-1] == self.vocab.EOS:
                    completed.append(beam)
                    k -= 1
                    continue

                # 遍历最好的k个候选集序列.
                for can in self.best_k(beam, k, encoder_output, x_padding_masks, x, torch.max(len_oovs)):
                    # 利用小顶堆来维护一个top_k的candidates.
                    # 小顶堆的值以当前序列的得分为准, 顺便也把候选集的id和候选集本身存储起来.
                    add2heap(topk, (can.seq_score(), id(can), can), k)

            # 当前候选集是堆元素的index=2的值can.
            curr = [items[2] for items in topk]
            # 候选集数量已经达到搜索宽度的时候, 停止搜索.
            if len(completed) == beam_size:
                break

        # 将最后产生的候选集追加进completed中.
        completed += curr

        # 按照得分进行降序排列, 取分数最高的作为当前解码结果序列.
        result = sorted(completed, key=lambda x: x.seq_score(), reverse=True)[0].tokens
        return result


    @timer(module='doing prediction')
    def predict(self, text, tokenize=True, beam_search=False):
        # 很重要的一个参数是将beam_search设置为True
        if isinstance(text, str) and tokenize:
            text = list(jieba.cut(text))
        # 做模型所需的若干张量的初始化操作
        x, oov = source2ids(text, self.vocab)
        x = torch.tensor(x).to(self.DEVICE)
        len_oovs = torch.tensor([len(oov)]).to(self.DEVICE)
        x_padding_masks = torch.ne(x, 0).byte().float()

        # 采用Beam search策略进行解码
        if beam_search:
            summary = self.beam_search(x.unsqueeze(0),
                                       max_sum_len=config.max_dec_steps,
                                       beam_size=config.beam_size,
                                       len_oovs=len_oovs,
                                       x_padding_masks=x_padding_masks)
        # 采用贪心策略进行解码
        else:
            summary = self.greedy_search(x.unsqueeze(0),
                                         max_sum_len=config.max_dec_steps,
                                         len_oovs=len_oovs,
                                         x_padding_masks=x_padding_masks)

        # 将数字化ids张量转换为自然语言文本
        summary = outputids2words(summary, oov, self.vocab)

        # 在模型摘要结果中删除掉<SOS>, <EOS>字符.
        return summary.replace('<SOS>', '').replace('<EOS>', '').strip() 