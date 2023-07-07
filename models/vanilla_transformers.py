import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor
from einops import rearrange
from hydra import initialize, compose
import omegaconf


class FFN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int, num_layers:int, dropout_p:float=0.1):
        super(FFN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_p))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            
        layers.append(nn.Linear(hidden_size, output_size))

        self.ffn = nn.Sequential(*layers)

    def forward(self, x:Tensor):
        return self.ffn(x)

# Single self-attention head
# class AttentionHead(nn.Module):
#     def __init__(self, emb_dim:int):
#         super(AttentionHead, self).__init__()
#         self.Q = nn.Linear(emb_dim, emb_dim, bias=False)
#         self.K = nn.Linear(emb_dim, emb_dim, bias=False)
#         self.V = nn.Linear(emb_dim, emb_dim, bias=False)
#         self.V_dim = emb_dim
    
#     def forward(self, x:Tensor):
#         q = self.Q(x)
#         k = self.K(x)
#         v = self.V(x)
#         k_transpose = rearrange(k, 'b s e -> b e s') #b is batch size, s is sequence length, e is embedding size
#         q_k_transpose = torch.bmm(q, k_transpose)
#         q_k_softmax = F.softmax(q_k_transpose/(self.V_dim**0.5), dim=2)
#         z = torch.bmm(q_k_softmax, v)
#         return z



    
# class MultiHeadAttention(nn.Module):
#     def __init__(self, emb_dim:int, num_attention_heads:int):
#         super(MultiHeadAttention, self).__init__()        
#         self.attention_heads = nn.ModuleList()
#         self.linear = nn.Linear(num_attention_heads*emb_dim, emb_dim)

#         for _ in range(num_attention_heads):
#             self.attention_heads.append(AttentionHead(emb_dim))
#     def forward(self, x:Tensor):
#         z = []
#         for i in range(len(self.attention_heads)):
#             z.append(self.attention_heads[i](x))
#         z = torch.cat(z, dim=2)
#         z = self.linear(z)
#         return z
class MultiHeadAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    @staticmethod
    def get_default_cfg():
        cfg = omegaconf.OmegaConf.load("conf/BERT.yaml")
        return cfg

    def __init__(self):
        super().__init__()

        config = self.get_default_cfg()
        assert config.model.emb_dim % config.model.num_attention_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.model.emb_dim, 3 * config.model.emb_dim)
        # output projection
        self.c_proj = nn.Linear(config.model.emb_dim, config.model.emb_dim)
        # regularization
        # self.attn_dropout = nn.Dropout(config.attn_pdrop)
        # self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.model.num_attention_heads
        self.n_embd = config.model.emb_dim

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C git // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        # y = self.resid_dropout(self.c_proj(y))
        return y


class EncoderBlock(nn.Module):
    def __init__(self, emb_dim:int, num_attention_heads:int, hidden_size:int=512, num_layers:int=4, dropout_p=0.1):
        super(EncoderBlock, self).__init__()
        self.ffn = FFN(emb_dim, hidden_size, emb_dim, num_layers, dropout_p)        
        self.multi_head_attention = MultiHeadAttention()
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
    
    def forward(self, x:Tensor):     
        z = self.multi_head_attention(x)
        z = self.layer_norm1(x + z) #residual connection + layer normalization
        z1 = self.ffn(z)
        z = self.layer_norm2(z + z1)
        return z
    
class Encoder(nn.Module):
    def __init__(self, emb_dim:int, vocab_size:int, num_attention_heads=5, num_encoder_blocks=3, max_seq_len=512, hidden_size=512, num_layers=4, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential()
        self.embedding = nn.Sequential(nn.Embedding(vocab_size, emb_dim), nn.Dropout(dropout_p))
        self.pos_encoding = PositionalEncoding(emb_dim, dropout_p, max_seq_len)
        for _ in range(num_encoder_blocks):
            self.encoder.append(EncoderBlock(emb_dim, num_attention_heads, hidden_size, num_layers, dropout_p))

    def forward(self, x:Tensor):
        x = self.embedding(x)
        x = rearrange(x, 'b s e -> s b e') # rearrange to fit positional encoding
        x = self.pos_encoding(x)
        x = rearrange(x, 's b e -> b s e')
        x = self.encoder(x)
        return x

    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



