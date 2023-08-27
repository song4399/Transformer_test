
import torch
import numpy as np
import io
from Transformer_test import Transformer_Model, PositionalEncoding
from hyperparams import Hyperparams as hp
from data_load import (
    load_vocab, load_cn_vocab,load_en_vocab, load_train_data, load_test_data)

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the CUDA")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

### 加载模型
cn2idx, idx2cn = load_cn_vocab() # 单词索引字典
en2idx, idx2en = load_en_vocab()
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

### 加载模型参数
path = 'D:\\About_RADAR\\About_DeepLearning\\Bi_She\\Transformer_code\\models\\model_epoch_70.pth'
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
model.to(device)
model.eval()

### 加载测试数据X: cn    # Y: en
X, Y, Sources, _ = load_test_data() # 加载训练集句子，并转换为padding后的索引序列

### 测试一个句子
x = torch.LongTensor(np.expand_dims(X[18,:], axis=0)).to(device)
y = torch.LongTensor(np.expand_dims(Y[18,:], axis=0)).to(device)
print(y[:,1:])
decoder_input = y[:, 0:1] # 起始字符
predictions = [] # 存储每次预测的结果
start_tocken = y[:, 0:1] # 常量起始字符
with torch.no_grad():
    for pos in range(hp.maxlen):
        probs = model(x, decoder_input)
        preds = torch.argmax(probs, -1) 
        predictions = preds
        decoder_input = torch.cat((start_tocken,preds),dim=1)

print(predictions)

### 具体接口
# input_seq = '普京 指出 两 国 高层 领导人 保持 定期 会 晤对 发展 和 深化 俄 中 战略 协作 伙伴 关系 至关 重要'
input_seq = '今天 此间 举行 的 外交部 新闻 发布会 上 章启月 说 中国 一直 十分 关注 阿富汗 的 有关 问题'
print(input_seq)
cn2idx, idx2cn = load_vocab("cn") # 返回索引映射 每个中文/英文单词对应的索引值
en2idx, idx2en = load_vocab("en") # 返回索引映射 每个中文/英文单词对应的索引值
x = [cn2idx.get(word, 1) for word in (input_seq + " <eos>").split()]  
# 1: OOV, </S>: End of Text # 语句末尾加上一个空格和末尾标记</S>，根据空格分割转换为列表，再获取对应的单词的索引，组成x
assert len(x) <= hp.maxlen-1, "句子太长！" # 如果最大长度在范围内，就接收这个序列
X = np.zeros([1, hp.maxlen], np.int32) # 创建空的数组，大小为（样本数量*最大长度）
X = np.lib.pad(x, [0, hp.maxlen - len(x)], "constant", constant_values=(2, 2)  )
# print(X)

x = torch.LongTensor(np.expand_dims(X, axis=0)).to(device)
decoder_input = torch.tensor([[0]]).to(device) # 起始字符
predictions = [] # 存储每次预测的结果
start_tocken = torch.tensor([[0]]).to(device) # 常量起始字符
with torch.no_grad():
    for pos in range(hp.maxlen):
        probs = model(x, decoder_input)
        preds = torch.argmax(probs, -1) 
        predictions = preds
        decoder_input = torch.cat((start_tocken,preds),dim=1).to(device)

predictions = predictions[0].tolist()
output = ''
for idx in predictions:
    y = idx2en.get(idx)
    if y=='<eos>':
        break
    output += y + ' '

print(output)