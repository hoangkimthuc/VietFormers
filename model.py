import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils import getPositionEncoding


class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_p=0.1):
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

    def forward(self, x):
        return self.ffn(x)

# Single self-attention head
class AttentionHead(nn.Module):
    def __init__(self, Q_dim, K_dim, V_dim):
        super(AttentionHead, self).__init__()
        self.Q = nn.Linear(Q_dim, Q_dim, bias=False)
        self.K = nn.Linear(K_dim, K_dim, bias=False)
        self.V = nn.Linear(V_dim, V_dim, bias=False)
        self.V_dim = V_dim
    
    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        k_transpose = rearrange(k, 'b s e -> b e s') #b is batch size, s is sequence length, e is embedding size
        q_k_transpose = torch.bmm(q, k_transpose)
        q_k_softmax = F.softmax(q_k_transpose/(self.V_dim**0.5), dim=2)
        z = torch.bmm(q_k_softmax, v)
        return z
    
class MultiHeadAttention(nn.Module):
    def __init__(self, Q_dim, K_dim, V_dim, num_attention_heads):
        super(MultiHeadAttention, self).__init__()        
        self.attention_heads = nn.ModuleList()
        self.linear = nn.Linear(num_attention_heads*V_dim, V_dim)

        for _ in range(num_attention_heads):
            self.attention_heads.append(AttentionHead(Q_dim, K_dim, V_dim))
    def forward(self, x):
        z = []
        for i in range(len(self.attention_heads)):
            z.append(self.attention_heads[i](x))
        z = torch.cat(z, dim=2)
        z = self.linear(z)
        return z

class EncoderBlock(nn.Module):
    def __init__(self, Q, K, V, num_heads, hidden_size=512, num_layers=4, dropout_p=0.1):
        super(EncoderBlock, self).__init__()
        self.ffn = FFN(V, hidden_size, V, num_layers, dropout_p)        
        self.multi_head_attention = MultiHeadAttention(Q, K, V, num_heads)
        self.layer_norm1 = nn.LayerNorm(V)
        self.layer_norm2 = nn.LayerNorm(V)
    
    def forward(self, x):     
        z = self.multi_head_attention(x)
        z = self.layer_norm1(x + z) #residual connection + layer normalization
        z1 = self.ffn(z)
        z = self.layer_norm2(z + z1)
        return z

class Encoder(nn.Module):
    def __init__(self, Q, K, V, num_attention_heads, num_encoder_blocks, max_len=512, hidden_size=512, num_layers=4, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential()
        for _ in range(num_encoder_blocks):
            self.encoder.append(EncoderBlock(Q, K, V, num_attention_heads, hidden_size, num_layers, dropout_p))
        self.pos_encoding = torch.tensor(getPositionEncoding(seq_len=max_len, d=V)).float()
        self.attention_mask = None

    def forward(self, x):
        z = x + self.pos_encoding
        z = self.encoder(z)
        return z



