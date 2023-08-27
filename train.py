import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import io

from hyperparams import Hyperparams as hp
from data_load import (
    get_batch_indices, load_cn_vocab,load_en_vocab, load_train_data)
from Transformer_test import Transformer_Model, PositionalEncoding

# Train Config
n_epochs = 70 # hp.num_epochs
batch_size = hp.batch_size
lr = hp.lr
print_interval = 40 # 隔x个batch打印一次
check_frequency = hp.check_frequency # 参数保存频率

maxlen = hp.maxlen

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the CUDA")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

class label_smoothing(nn.Module):
    def __init__(self, epsilon=0.1):
        super(label_smoothing, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs):
        K = inputs.size()[-1]
        return ((1 - self.epsilon) * inputs) + (self.epsilon / K)
label_smoothing = label_smoothing()

def main():
    cn2idx, idx2cn = load_cn_vocab() # 单词索引字典
    en2idx, idx2en = load_en_vocab()
    # X: cn    # Y: en
    X, Y, Sources, _ = load_train_data() # 加载训练集句子，并转换为padding后的索引序列

    model = Transformer_Model(src_voc_len=len(cn2idx), # 单词表大小
                              src_emb_dim=hp.emb_dim,
                              tgt_voc_len=len(en2idx),
                              tgt_emb_dim=hp.emb_dim,
                              PAD_ID=hp.PAD_ID,
                              maxlen=hp.maxlen,
                              num_head=hp.num_heads,
                              num_hiddens=hp.num_hiddens,
                              hidden_channel=hp.hidden_channel,
                              output_channel=hp.emb_dim,
                              num_encoder_layers=hp.num_layer,
                              num_decoder_layers=hp.num_layer,
                              dropout=hp.dropout_rate,
                              layer_norm_eps=1e-5,
                              device=device
    )
    torch.save(model.enc_position_enc.state_dict(), 'models\\enc_position_enc.pth')
    torch.save(model.dec_position_enc.state_dict(), 'models\\dec_position_enc.pth')

    if True:
        path = 'D:\\About_RADAR\\About_DeepLearning\\Bi_She\\Transformer_code\\models\\model_epoch_40.pth'
        with open(path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        state_dict = torch.load(buffer)
        del state_dict["enc_position_enc.pe"]
        del state_dict["dec_position_enc.pe"]
        # 创建新的位置编码模块
        enc_position_enc = PositionalEncoding(hp.emb_dim, device=device)
        dec_position_enc = PositionalEncoding(hp.emb_dim, device=device)

        # 加载位置编码权重到新模块
        enc_position_enc.load_state_dict(torch.load('D:\\About_RADAR\\About_DeepLearning\\Bi_She\\Transformer_code\\models\\enc_position_enc.pth'))
        dec_position_enc.load_state_dict(torch.load('D:\\About_RADAR\\About_DeepLearning\\Bi_She\\Transformer_code\\models\\dec_position_enc.pth'))

        # 替换模型中原始的位置编码模块
        model.load_state_dict(state_dict, strict=False)
        model.enc_position_enc = enc_position_enc
        model.dec_position_enc = dec_position_enc

    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    citerion = nn.CrossEntropyLoss(ignore_index=hp.PAD_ID).to(device) # ignore_index索引值在计算损失时被忽略
    tic = time.time()
    num_batch = len(X) // batch_size # 一共多少个batch

    for epoch in range(40+1, n_epochs+1):
        current_batches = 0
        for index, _ in get_batch_indices(len(X), batch_size):
            x_batch = torch.LongTensor(X[index]).to(device) # 一个batch的数据，句子索引，输入到encoder里面, index是对的, X也是对的
            y_batch = torch.LongTensor(Y[index]).to(device) # 输入到encoder里面的token[n]，正常的，和解码器输入对应的

            # 得到概率分布
            probs = model(x_batch, y_batch) # 输出(batch_size, tgt_voc_len)并且Softmax之后

            # 得到预测序列
            preds = torch.argmax(probs, -1) # 先将y_hat转换为索引序列。没有开始字符，长度token[n] torch.Size([64, 50])
            
            # label是需要和预测序列去对比来确定精度的目标
            label = torch.cat((y_batch[:, 1:maxlen], torch.full((y_batch.size()[0], 1), fill_value=2, device=device) ), dim=1)
            
            # istarget是判断哪些是目标值，不能为padding
            istarget = (1.0 - torch.cat((y_batch[:, 1:maxlen], torch.full((y_batch.size()[0], 1), fill_value=2, device=device) ), dim=1).eq(2.0).float() ).view(-1) # 判断哪些是目标样本，不能=2
            
            # 计算精度            
            acc = torch.sum(preds.eq(label).float().view(-1)*istarget) / torch.sum(istarget) # 所有不为PAD的索引都是需要正确预测的单词
            # print(preds[1,:])
            # print(label[1,:])
            # 计算Loss
            # loss = citerion(probs.view(-1, len(en2idx)), label.view(-1) )
            # 另一种计算loss
            y_onehot = torch.zeros(batch_size * hp.maxlen, len(en2idx), device=device)
            y_onehot = y_onehot.scatter_(1,label.view(-1,1).data,1)
            # 2是沿第几个轴，label.view(-1,1).data是填充索引，1是填充源
            y_smoothed = label_smoothing(y_onehot)
            loss = torch.sum(-torch.sum(y_onehot * torch.log(probs.view(-1, len(en2idx))), dim=-1) )
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播，得到梯度
            optimizer.step() # 梯度更新

            if (current_batches % print_interval == 0 
                or current_batches == 0 
                or current_batches == num_batch): # 打印间隔，每隔多少个batch打印一次，每个batch count+1
                toc = time.time()
                interval = toc - tic
                minutes = int(interval // 60)
                seconds = int(interval % 60)
                print(f'{minutes:02d}:{seconds:02d} Epoch: {epoch}, '
                      f'batch: {current_batches}/{num_batch} ({(current_batches/num_batch):.2%}) '
                      f'loss: {loss.item()} acc: {acc.item()}')
                print('probs[0,:]')
                print(preds[0,:])
                print(label[0,:])
            current_batches += 1
            # break
        if epoch % check_frequency == 0 or epoch == n_epochs: # 每隔一段保存参数
            checkpoint_path = hp.model_dir + "/model_epoch_%02d" % epoch + ".pth"
            torch.save(model.state_dict(), checkpoint_path) # 保存模型的参数


if __name__ == '__main__':
    main()