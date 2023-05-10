import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, Q, K, V):
        super(AttentionHead, self).__init__()
        self.Q = nn.Linear(Q, Q)
        self.K = nn.Linear(K, K)
        self.V = nn.Linear(V, V)
    
    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        k_transpose = k.transpose(1, 0)
        q_k_transpose = torch.matmul(q, k_transpose)        
        q_k_softmax = F.softmax(q_k_transpose)/torch.sqrt(torch.tensor(k.shape[1]).float())
        z = torch.matmul(q_k_softmax, v)
        return z

class Encoder(nn.Module):
    def __init__(self, Q, K, V, heads, hidden_size=512, num_layers=4, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.ffn = FFN(V, hidden_size, V, num_layers, dropout_p)
        self.heads = heads
        self.attention_heads = nn.ModuleList()
        for _ in range(heads):
            self.attention_heads.append(AttentionHead(Q, K, V))   #create a list of attention heads (multi-head attention)
        self.linear = nn.Linear(heads*V, V)
        self.layer_norm1 = nn.LayerNorm(V)
        self.layer_norm2 = nn.LayerNorm(V)
    
    def forward(self, x):
        z = []
        for i in range(self.heads):
            z.append(self.attention_heads[i](x)) #apply attention heads
        z = torch.cat(z, dim=1)
        z = self.linear(z)
        z = self.layer_norm1(x + z) #residual connection + layer normalization
        z1 = self.ffn(z)
        z = self.layer_norm2(z + z1)
        return z

class MultiEncoder(nn.Module):
    def __init__(self, Q, K, V, heads, num_encoder, seq_len, hidden_size=512, num_layers=4, dropout_p=0.1):
        super(MultiEncoder, self).__init__()
        self.multiencoder = nn.ModuleList()
        self.num_encoder = num_encoder
        for _ in range(num_encoder):
            self.multiencoder.append(Encoder(Q, K, V, heads, hidden_size, num_layers, dropout_p))
        self.pos_encoding = torch.tensor(getPositionEncoding(seq_len=seq_len, d=V)).float()
        self.attention_mask = None

    def forward(self, x):
        z = x + self.pos_encoding
        for i in range(self.num_encoder):
            z = self.multiencoder[i](z)
        return z



