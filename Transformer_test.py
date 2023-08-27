import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import Tensor
from typing import Optional, Union, Callable
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from typing import Optional, Tuple
import math
# torch.manual_seed(4)
MY_INF = 2**32
# xavier_normal_正太分布 xavier_uniform_ 均匀分布

class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, attn_mask=False) -> Tensor:
        output = torch.bmm(Q, K.permute(0,2,1))
        output = output / (Q.size(-1) ** 0.5) 
        # 输出(batchsize*num_heads, seq_len, seq_len)
        
        # key_mask
        key_masks = torch.sign(torch.abs(torch.sum(K, dim=-1))) # PAD都是0
        # K是(batchsize*num_heads, seq_len, num_hiddens//num_heads) 
        # key_masks是(batch_size*num_heads, seq_len)
        # 如果单词索引是PAD_ID那K的那一行都是输出0，其他都不为0
        key_masks = torch.unsqueeze(key_masks, 1) 
        # 输出(batch_size*num_heads, 1, seq_len)
        key_masks = key_masks.repeat(1, Q.size(1), 1) 
        # 输出(batch_size*num_heads, seq_len, seq_len)
        padding = torch.ones(*output.size(), device=self.device) * (-(MY_INF) + 1)
        condition = key_masks.eq(0.0).float().to(self.device)
        output = padding * condition + output * (1.0 - condition) 
        # 输出(batchsize*num_heads, seq_len, seq_len)

        # attn_mask
        if attn_mask: # decoder mask
            diag_vals = torch.ones(*output[0, :, :].size(), device=self.device)
            tril = torch.tril(diag_vals, diagonal=0) # 下三角为1
            masks = torch.unsqueeze(tril, 0).repeat(output.size()[0], 1, 1) 
            # 输出(batchsize*num_heads, seq_len, seq_len)

            padding = torch.ones(*masks.size(), device=self.device) * (-(MY_INF) + 1)
            condition = masks.eq(0.0).float()
            output = padding * condition + output * (1.0 - condition)
        
        output = F.softmax(output, dim=-1) # 输出(batchsize*num_heads, seq_len, seq_len)
        output = self.dropout(output)
        output = torch.bmm(output, V) 
        # 输出(batchsize*num_heads, seq_len, num_hiddens//num_heads)

        return output


class MultiheadAttention(nn.Module): # 多头注意力类
    def __init__(self, query_size, key_size, value_size, num_hiddens, output_channel, 
                 num_heads, dropout=0.1, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__() 
        # 如果子类的构造函数需要传递参数给父类，则在init里面传递参数
        # 并且需要在父类中的init里面有声明
        self.device = device
        self.num_heads = num_heads # head的个数
        self.dropout = nn.Dropout(dropout)
        
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias, **factory_kwargs) 
        # (input_channel, output_channel)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias, **factory_kwargs)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias, **factory_kwargs)

        self.attention = SelfAttention(dropout, **factory_kwargs)
        
        self.W_o = nn.Linear(num_hiddens, output_channel, bias=bias, **factory_kwargs)

    def forward(self,query: Tensor, key: Tensor, value: Tensor, 
                attn_mask=False) -> Tuple[Tensor, Optional[Tensor]]:
        # q k v必须是(batchsize, seq_len, emb_dim) emb_dim: input_channel
        # Wq Wk Wv是(num_hiddens, emb_dim)
        # Linear是x*A^T A是权重矩阵
        # 输出QKV是(batchsize, seq_len, num_hiddens)
        Q = self.W_q(query) # 先经过线性层
        K = self.W_k(key)
        V = self.W_v(value)
        Q = F.relu(Q) # 再经过激活层
        K = F.relu(K)
        V = F.relu(V)
        assert Q.shape[2] % self.num_heads==0, f"{Q.shape[2]} 不能被 {self.num_heads} 整除"
        Q = Q.reshape(Q.shape[0], Q.shape[1], self.num_heads, -1) # 再进行reshape到多头
        K = K.reshape(K.shape[0], K.shape[1], self.num_heads, -1)
        V = V.reshape(V.shape[0], V.shape[1], self.num_heads, -1)

        Q = Q.permute(0,2,1,3) 
        # 输出(batchsize, num_heads, seq_len, num_hiddens//num_heads)
        K = K.permute(0,2,1,3)
        V = V.permute(0,2,1,3)

        Q = Q.reshape(-1, Q.shape[2], Q.shape[3]) 
        # 输出(batchsize*num_heads, seq_len, num_hiddens//num_heads)
        K = K.reshape(-1, K.shape[2], K.shape[3])
        V = V.reshape(-1, V.shape[2], V.shape[3])

        output = self.attention(Q,K,V, attn_mask) # 注意力输出
        # 输出(batchsize*num_heads, seq_len, num_hiddens//num_heads)

        output = output.reshape(-1, self.num_heads, output.shape[1], output.shape[2])
        # 输出(batchsize, num_heads, seq_len, num_hiddens//num_heads)
        output = output.permute(0,2,1,3) 
        # 输出(batchsize, seq_len, num_heads, num_hiddens//num_heads)
        output = output.reshape(output.shape[0], output.shape[1], -1) 
        # 输出(batchsize, seq_len, num_hiddens)
        output = self.W_o(output) # 输出(batchsize, seq_len, num_hiddens)

        return output

class FeedForward(nn.Module):
    def __init__(self, in_channel, hidden_channel, output_channel, eps=1e-6,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.dense1 = nn.Linear(in_channel, hidden_channel, **factory_kwargs)
        self.dense2 = nn.Linear(hidden_channel, output_channel, **factory_kwargs)
        self.norm = nn.LayerNorm(output_channel, eps=eps, **factory_kwargs)
        
    def forward(self, input)-> Tensor:
        output = self.dense1(input)
        output = F.relu(output)
        output = self.dense2(output)
        # Residual connection
        output += input
        # Layer normalization
        output = self.norm(output)
        return output

# 一个encoder层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_head, num_hiddens, 
                 hidden_channel, output_channel, dropout = 0.1, layer_norm_eps = 1e-5, 
                 device=None, dtype=None) -> Tensor:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        # 多头注意力模块
        self.self_attn = MultiheadAttention(query_size, key_size, value_size, num_hiddens, 
                                            output_channel, num_head, dropout, bias=True, 
                                            **factory_kwargs)
        # Feedforward model
        self.feedforward = FeedForward(output_channel, hidden_channel, output_channel, 
                                       eps=layer_norm_eps, **factory_kwargs)

        self.norm1 = nn.LayerNorm(output_channel, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(output_channel, eps=layer_norm_eps, **factory_kwargs)

    def forward(self, src: Tensor, ) -> Tensor:
        x = src # x: (batchsize, src_seq_len, src_emb_dim)
        x = self.norm1(x + (self.self_attn(x, x, x, attn_mask = False) ) )
        x = self.norm2(x + (self.feedforward(x) ) )

        return x

# 一个decoder模块
class TransformerDecoderLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_head, num_hiddens, 
                 hidden_channel, output_channel, dropout = 0.1, layer_norm_eps = 1e-5, 
                 device=None, dtype=None) -> Tensor:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(query_size, key_size, value_size, num_hiddens, 
                                            output_channel, num_head, dropout, bias=True, 
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(query_size, key_size, value_size, 
                                                 num_hiddens, output_channel, num_head, 
                                                 dropout, bias=True, **factory_kwargs)
        # Implementation of Feedforward model
        self.feedforward = FeedForward(output_channel, hidden_channel, output_channel, 
                                       eps=layer_norm_eps, **factory_kwargs)

        self.norm1 = nn.LayerNorm(output_channel, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(output_channel, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(output_channel, eps=layer_norm_eps, **factory_kwargs)

    def forward(self, tgt: Tensor, memory: Tensor, attn_mask=True ) -> Tensor:
        x = tgt # (batchsize, tgt_seq_len, tgt_emb_dim)
        x = self.norm1(x + (self.self_attn(x, x, x,attn_mask=attn_mask) ) )
        x = self.norm2(x + (self.multihead_attn(x, memory, memory, attn_mask=False) ) )
        x = self.norm3(x + (self.feedforward(x) ) )

        return x


class TransformerEncoder(nn.Module):
    # 初始化类，主要需要：输入encoder模块的实例，输入层数，输入norm类型
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) 
                                     for i in range(num_layers)]) # 将module复制N份

    def forward(self, src: Tensor) -> Tensor:
        output = src # 源seq
        for encoderlayer in self.layers: # 6层编码器
            output = encoderlayer(output)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) 
                                     for i in range(num_layers)])

    def forward(self, tgt: Tensor, memory: Tensor, attn_mask=True) -> Tensor:
        output = tgt
        for decoderlayer in self.layers:
            output = decoderlayer(output, memory, attn_mask=attn_mask)

        return output


class Transformer(nn.Module):
    def __init__(self, src_emb_dim, tgt_emb_dim, num_head=8, num_hiddens=512, 
                 hidden_channel=2048, output_channel=512, num_encoder_layers=6, 
                 num_decoder_layers=6, dropout=0.1, layer_norm_eps=1e-5, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()  # 调用父类的方法，继承父类的属性

        # encoder_layer 生成TransformerEncoderLayer类的实例
        encoder_layer = TransformerEncoderLayer(src_emb_dim, src_emb_dim, src_emb_dim, 
                                                num_head, num_hiddens, hidden_channel, 
                                                output_channel, dropout, layer_norm_eps, 
                                                **factory_kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        # decoder_layer
        decoder_layer = TransformerDecoderLayer(tgt_emb_dim, tgt_emb_dim, tgt_emb_dim, 
                                                num_head, num_hiddens, hidden_channel, 
                                                output_channel, dropout, layer_norm_eps,
                                                **factory_kwargs)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(self, src: Tensor, tgt: Tensor, tgt_mask=True,) -> Tensor:
        # encoder 输入的是（索引和位置嵌入）之后的源，存储最后一个编码器是输出
        memory = self.encoder(src)
        # decoder 输入是tgt源，输出是还没经过Linear处理的结果
        output = self.decoder(tgt, memory, attn_mask=tgt_mask)
        return output


class Embedding(nn.Module):
    def __init__(self, voc_len, emb_dim, zeros_pad=True, PAD_ID = None, scale=True, 
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.zeros_pad = zeros_pad
        if self.zeros_pad:
            self.lut = nn.Embedding(voc_len, emb_dim, padding_idx=PAD_ID, **factory_kwargs)
        else:
            self.lut = nn.Embedding(voc_len, emb_dim, **factory_kwargs)
        self.emb_dim = emb_dim  #表示embedding的维度
        self.scale = scale
 
    def forward(self, x):
        if self.scale:
            return self.lut(x) * (self.emb_dim**0.5)
        else:
            return self.lut(x)


class PositionalEncoding(nn.Module):  # 位置编码，用在输入层，每一层的开始，编码器，解码器开始
    def __init__(self, emb_dim, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.emb_dim = emb_dim
        self.device = device
 
    def forward(self, x: Tensor) -> Tensor:  
        # x应该是已经经过词编码的向量 (batchsize, seq_len, emb_dim)
        n_position_max=x.size(1)
        pe = torch.zeros(n_position_max, self.emb_dim , device=self.device)
        position = torch.arange(0., n_position_max, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., self.emb_dim , 2, device=self.device) / 
                             self.emb_dim  * (math.log(10000.0) ) )
        pe[:, 0::2] = torch.sin(position / div_term)    # 偶数列
        pe[:, 1::2] = torch.cos(position / div_term)    # 奇数列
        self.register_buffer('pe', pe)
        x = x + (self.pe[:, :x.size(-1)])
        return x


class Transformer_Model(nn.Module):
    def __init__(self, src_voc_len = 10000, src_emb_dim=512, tgt_voc_len = 10000, 
                 tgt_emb_dim=512, PAD_ID=2, maxlen = 50, num_head=8, num_hiddens=512, 
                 hidden_channel=2048, output_channel=512, num_encoder_layers=6, 
                 num_decoder_layers=6, dropout=0.1, layer_norm_eps=0.00001, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        ###### 使用传统Embedding的
        self.src_embed = Embedding(voc_len=src_voc_len, emb_dim=src_emb_dim, 
                                   PAD_ID=PAD_ID, **factory_kwargs)
        ###### 使用传统Embedding的
        self.tgt_embed = Embedding(voc_len=tgt_voc_len, emb_dim=tgt_emb_dim, 
                                   PAD_ID=PAD_ID, **factory_kwargs)

        self.enc_position_enc = PositionalEncoding(emb_dim=src_emb_dim, **factory_kwargs)
        self.dec_position_enc = PositionalEncoding(emb_dim=tgt_emb_dim, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(src_emb_dim, tgt_emb_dim, num_head, num_hiddens,
                                       hidden_channel, output_channel, num_encoder_layers, 
                                        num_decoder_layers, dropout, layer_norm_eps, 
                                        **factory_kwargs)
        
        self.tgt_out = nn.Linear(output_channel, tgt_voc_len, bias=False, **factory_kwargs) 
        #(输入维度，输出维度) 创建一个(输出维度，输入维度)的权重矩阵A，之后x*A^T

    def forward(self, src_seq: Tensor, tgt_seq: Tensor) -> Tensor:
        input_src = (self.dropout(self.enc_position_enc(self.src_embed(src_seq))))
        input_tgt = (self.dropout(self.dec_position_enc(self.tgt_embed(tgt_seq))))

        decoder_output = self.transformer(input_src, input_tgt, tgt_mask=True)
        output = F.softmax(self.tgt_out(decoder_output),-1)
        return output


# ##### TEST
# d_model = 4
# maxlen = 3
# src_seq = torch.LongTensor([[1,2,0],[1,3,0]])
# tgt_seq = torch.LongTensor([[1,3,0],[2,1,0]])
# print(src_mask)
# print(tgt_mask)
# print(src_key_padding_mask)
# print(tgt_key_padding_mask)
# src_embedding = nn.Embedding(num_embeddings=d_model, embedding_dim=d_model,padding_idx=0)
# tgt_embedding = nn.Embedding(num_embeddings=d_model, embedding_dim=d_model,padding_idx=0)
# position_enc = PositionalEncoding(d_model, n_position=maxlen)
# Model_Transformer = Transformer(d_model=d_model, nhead=4)
# src_in = position_enc(src_embedding(src_seq))
# tgt_in = position_enc(tgt_embedding(tgt_seq))
# output = Model_Transformer(src_in, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask,
#                 memory_mask=None, src_key_padding_mask=src_key_padding_mask,
#                 tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=None)
# print(output)
# ##### TEST OVER!

# ##### TEST MODEL
# src_seq = torch.LongTensor([[1,3,4,5,6,7,5,2],[1,4,6,5,4,2,2,2]]).to('cuda:0')
# tgt_seq = torch.LongTensor([[1,3,6,5,4,2,2],[1,3,5,6,4,3,5]]).to('cuda:0')
# Model = Transformer_Model(
#     src_voc_len=10000,
#     src_emb_dim=512,
#     tgt_voc_len=12000,
#     tgt_emb_dim=512,
#     PAD_ID=2,
#     maxlen=8,
#     num_head=16,
#     num_hiddens=1024, # 自注意力层隐藏维度
#     hidden_channel=2048, # FF层隐藏维度
#     output_channel=512, # 输入输出维度 需要和src_emb_dim tgt_emb_dim相等
#     num_encoder_layers=6,
#     num_decoder_layers=6,
#     dropout=0.4,
#     layer_norm_eps=1e-6,
#     device='cuda:0'
# )
# # Model.eval()
# output = Model(src_seq, tgt_seq)
# print(output)
# ##### TEST MODEL OVER!
