import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor
from einops import rearrange

class FFN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int, num_layers:int, ffn_pdrop:float):
        super(FFN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(ffn_pdrop))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(ffn_pdrop))
            
        layers.append(nn.Linear(hidden_size, output_size))

        self.ffn = nn.Sequential(*layers)

    def forward(self, x:Tensor):
        return self.ffn(x)

class MultiHeadAttention(nn.Module):  
    def __init__(self, emb_dim:int, num_attention_heads:int, attn_pdrop:float, linear_proj_pdrop:float):
        super().__init__()

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(emb_dim, 3 * num_attention_heads*emb_dim, bias=False)
        # output projection
        self.c_proj = nn.Linear(num_attention_heads*emb_dim, emb_dim)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.linear_proj_dropout = nn.Dropout(linear_proj_pdrop)        
        self.n_head = num_attention_heads
        self.n_embd = emb_dim

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_head*self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C ).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C ).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C ).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head*C) # re-assemble all head outputs side by side

        # output projection
        y = self.linear_proj_dropout(self.c_proj(y))
        return y


class EncoderBlock(nn.Module):
    def __init__(self, emb_dim:int, num_attention_heads:int, hidden_size:int, num_layers:int, ffn_pdrop:float, attn_pdrop:float, linear_proj_pdrop:float):
        super(EncoderBlock, self).__init__()
        self.ffn = FFN(emb_dim, hidden_size, emb_dim, num_layers, ffn_pdrop)        
        self.multi_head_attention = MultiHeadAttention(emb_dim=emb_dim, num_attention_heads=num_attention_heads, attn_pdrop=attn_pdrop, linear_proj_pdrop=linear_proj_pdrop)
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
    
    def forward(self, x:Tensor):     
        z = self.multi_head_attention(x)
        z = self.layer_norm1(x + z) #residual connection + layer normalization
        z1 = self.ffn(z)
        z = self.layer_norm2(z + z1)
        return z
    
class Encoder(nn.Module):
    def __init__(self, emb_dim:int, num_attention_heads:int, 
                 hidden_size:int, num_layers:int, ffn_pdrop:float, 
                 attn_pdrop:float, linear_proj_pdrop:float, 
                 pos_emb_pdrop:float, emb_pdrop:float,
                 vocab_size:int, max_seq_len:int, num_encoder_blocks:int):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.emb_drop = nn.Dropout(emb_pdrop)
        self.pos_encoding = PositionalEncoding(emb_dim, pos_emb_pdrop, max_seq_len)
        for _ in range(num_encoder_blocks):
            self.encoder.append(EncoderBlock(emb_dim=emb_dim, num_attention_heads=num_attention_heads, 
                                   hidden_size=hidden_size, num_layers=num_layers, 
                                   ffn_pdrop=ffn_pdrop, attn_pdrop=attn_pdrop, 
                                   linear_proj_pdrop=linear_proj_pdrop))
    def forward(self, x:Tensor):
        x = self.emb_drop(self.embedding(x))
        x = rearrange(x, 'b s e -> s b e') # rearrange to fit positional encoding
        x = self.pos_encoding(x)
        x = rearrange(x, 's b e -> b s e')
        x = self.encoder(x)
        return x

    
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, pos_emb_pdrop: float, max_seq_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=pos_emb_pdrop)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(max_seq_len, 1, emb_dim)
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



