import torch
import torch.nn as nn
import torch.nn.functional as F
# Write the self-attention model here
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

class MultiAttention(nn.Module):
    def __init__(self, Q, K, V, heads):
        super(MultiAttention, self).__init__()
        self.heads = heads
        self.attention_heads = nn.ModuleList()
        for _ in range(heads):
            self.attention_heads.append(AttentionHead(Q, K, V))
        self.linear = nn.Linear(heads*V, V)
    
    def forward(self, x):
        z = []
        for i in range(self.heads):
            z.append(self.attention_heads[i](x))
        z = torch.cat(z, dim=1)
        z = self.linear(z)
        return z

