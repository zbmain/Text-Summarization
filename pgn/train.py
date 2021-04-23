# 导入系统工具包
import pickle
import os
import sys
# 设置项目的root路径, 方便后续相关代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入项目中用到的工具包
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tensorboardX import SummaryWriter

# 导入项目中自定义的代码文件, 类, 函数等
from src.model import PGN
from utils import config
from src.evaluate import evaluate
from utils.dataset import PairDataset, collate_fn, SampleDataset
from utils.func_utils import ScheduledSampler, config_info


# 编写训练模型的主逻辑函数.
def train(dataset, val_dataset, v, start_epoch=0):
    DEVICE = config.DEVICE

    # 实例化PGN类对象并移动到GPU上(CPU).
    model = PGN(v)
    model.to(DEVICE)

    print("loading data......")
    train_data = SampleDataset(dataset.pairs, v)
    val_data = SampleDataset(val_dataset.pairs, v)

    print("initializing optimizer......")

    # 定义模型训练的优化器.
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 定义训练集的数据迭代器(这里用到了自定义的collate_fn以服务于PGN特殊的数据结构).
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)

    # 验证集上的损失值初始化为一个大整数.
    val_losses = 10000000.0

    # SummaryWriter: 为服务于TensorboardX写日志的可视化工具.
    writer = SummaryWriter(config.log_path)

    num_epochs =  len(range(start_epoch, config.epochs))


    # ---------------------------------------------------------------------------------------
    # 下面的代码是6.4小节添加的关于训练策略Scheduled sampling的优化代码.
    # 工具函数ScheduledSampler()根据当前模型训练epoch的推进, 来决定Teacher-forcing的策略选择
    scheduled_sampler = ScheduledSampler(num_epochs)
    if config.scheduled_sampling:
        print('scheduled_sampling mode.')
    # ---------------------------------------------------------------------------------------


    with tqdm(total=config.epochs) as epoch_progress:
        for epoch in range(start_epoch, config.epochs):
            # 每一个epoch之前打印模型训练的相关配置信息.
            print(config_info(config))

            # 初始化每一个batch损失值的存放列表
            batch_losses = []
            num_batches = len(train_dataloader)


            # ----------------------------------------------------------------------------
            # 下面的代码是6.4小节添加的关于训练策略Scheduled sampling的优化代码.
            # 调用工具函数决定是否使用Teacher-forcing策略.
            if config.scheduled_sampling:
                teacher_forcing = scheduled_sampler.teacher_forcing(epoch - start_epoch)
            else:
                teacher_forcing = True
            print('teacher_forcing = {}'.format(teacher_forcing))
            # ----------------------------------------------------------------------------


            with tqdm(total=num_batches//100) as batch_progress:
                for batch, data in enumerate(tqdm(train_dataloader)):
                    x, y, x_len, y_len, oov, len_oovs = data
                    assert not np.any(np.isnan(x.numpy()))

                    # 如果配置有GPU, 则加速训练
                    if config.is_cuda:
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)
                        x_len = x_len.to(DEVICE)
                        len_oovs = len_oovs.to(DEVICE)

                    # 设置模型进入训练模式(参数参与反向传播和更新)
                    model.train()

                    # "老三样"中的第一步: 梯度清零
                    optimizer.zero_grad()
                    # 调用模型进行训练并返回损失值
                    loss = model(x, x_len, y,
                                 len_oovs, batch=batch,
                                 num_batches=num_batches,
                                 teacher_forcing=teacher_forcing)

                    batch_losses.append(loss.item())

                    # "老三样"中的第二步: 反向传播
                    loss.backward()

                    # 为防止梯度爆炸(gradient explosion)而进行梯度裁剪.
                    clip_grad_norm_(model.encoder.parameters(), config.max_grad_norm)
                    clip_grad_norm_(model.decoder.parameters(), config.max_grad_norm)
                    clip_grad_norm_(model.attention.parameters(), config.max_grad_norm)

                    # "老三样"中的第三步: 参数更新
                    optimizer.step()

                    # 每隔100个batch记录一下损失值信息.
                    if (batch % 100) == 0:
                        batch_progress.set_description(f'Epoch {epoch}')
                        batch_progress.set_postfix(Batch=batch, Loss=loss.item())
                        batch_progress.update()
                        # 向tensorboard中写入损失值信息.
                        writer.add_scalar(f'Average loss for epoch {epoch}',
                                           np.mean(batch_losses),
                                           global_step=batch)

            # 将一个轮次中所有batch的平均损失值作为这个epoch的损失值.
            epoch_loss = np.mean(batch_losses)

            epoch_progress.set_description(f'Epoch {epoch}')
            epoch_progress.set_postfix(Loss=epoch_loss)
            epoch_progress.update()

            # 结束每一个epoch训练后, 直接在验证集上跑一下模型效果
            avg_val_loss = evaluate(model, val_data, epoch)

            print('training loss:{}'.format(epoch_loss), 'validation loss:{}'.format(avg_val_loss))

            # 更新更小的验证集损失值evaluating loss.
            if (avg_val_loss < val_losses):
                torch.save(model.encoder, config.encoder_save_name)
                torch.save(model.decoder, config.decoder_save_name)
                torch.save(model.attention, config.attention_save_name)
                torch.save(model.reduce_state, config.reduce_state_save_name)
                torch.save(model.state_dict(), config.model_save_path)
                val_losses = avg_val_loss

                # 将更小的损失值写入文件中
                with open(config.losses_path, 'wb') as f:
                    pickle.dump(val_losses, f)

    writer.close()

if __name__ == "__main__":
    # Prepare dataset for training.
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE: ', DEVICE)

    # 将构建数据集的路径改为config.train_data_path, 具体采用单词替换后的训练集
    dataset = PairDataset(config.train_data_path,
                          max_enc_len=config.max_enc_len,
                          max_dec_len=config.max_dec_len,
                          truncate_enc=config.truncate_enc,
                          truncate_dec=config.truncate_dec)
    val_dataset = PairDataset(config.val_data_path,
                              max_enc_len=config.max_enc_len,
                              max_dec_len=config.max_dec_len,
                              truncate_enc=config.truncate_enc,
                              truncate_dec=config.truncate_dec)

    vocab = dataset.build_vocab(embed_file=config.embed_file)

    train(dataset, val_dataset, vocab, start_epoch=0)