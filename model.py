import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor
from einops import rearrange


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
class AttentionHead(nn.Module):
    def __init__(self, Q_dim:int, K_dim:int, V_dim:int):
        super(AttentionHead, self).__init__()
        self.Q = nn.Linear(Q_dim, Q_dim, bias=False)
        self.K = nn.Linear(K_dim, K_dim, bias=False)
        self.V = nn.Linear(V_dim, V_dim, bias=False)
        self.V_dim = V_dim
    
    def forward(self, x:Tensor):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        k_transpose = rearrange(k, 'b s e -> b e s') #b is batch size, s is sequence length, e is embedding size
        q_k_transpose = torch.bmm(q, k_transpose)
        q_k_softmax = F.softmax(q_k_transpose/(self.V_dim**0.5), dim=2)
        z = torch.bmm(q_k_softmax, v)
        return z
    
class MultiHeadAttention(nn.Module):
    def __init__(self, Q_dim:int, K_dim:int, V_dim:int, num_attention_heads:int):
        super(MultiHeadAttention, self).__init__()        
        self.attention_heads = nn.ModuleList()
        self.linear = nn.Linear(num_attention_heads*V_dim, V_dim)

        for _ in range(num_attention_heads):
            self.attention_heads.append(AttentionHead(Q_dim, K_dim, V_dim))
    def forward(self, x:Tensor):
        z = []
        for i in range(len(self.attention_heads)):
            z.append(self.attention_heads[i](x))
        z = torch.cat(z, dim=2)
        z = self.linear(z)
        return z

class EncoderBlock(nn.Module):
    def __init__(self, Q_dim:int, K_dim:int, V_dim:int, num_heads:int, hidden_size=512, num_layers=4, dropout_p=0.1):
        super(EncoderBlock, self).__init__()
        self.ffn = FFN(V_dim, hidden_size, V_dim, num_layers, dropout_p)        
        self.multi_head_attention = MultiHeadAttention(Q_dim, K_dim, V_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(V_dim)
        self.layer_norm2 = nn.LayerNorm(V_dim)
    
    def forward(self, x:Tensor):     
        z = self.multi_head_attention(x)
        z = self.layer_norm1(x + z) #residual connection + layer normalization
        z1 = self.ffn(z)
        z = self.layer_norm2(z + z1)
        return z
    
class Encoder(nn.Module):
    def __init__(self, Q_dim:int, K_dim:int, V_dim:int, vocab_size:int=50000, num_attention_heads=5, num_encoder_blocks=3, max_len=512, hidden_size=512, num_layers=4, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential()
        self.embedding = nn.Embedding(vocab_size, V_dim)
        self.pos_encoding = PositionalEncoding(V_dim, dropout_p, max_len)
        for _ in range(num_encoder_blocks):
            self.encoder.append(EncoderBlock(Q_dim, K_dim, V_dim, num_attention_heads, hidden_size, num_layers, dropout_p))

    def forward(self, x:Tensor):
        x = self.embedding(x)
        x = rearrange(x, 'b s e -> s b e')
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



