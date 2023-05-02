import torch
from model import AttentionHead, MultiAttention

# Test the above class
def test_attention_head():
    Q, K, V = 10, 10, 10
    x = torch.randn(2, 10)
    attention_head = AttentionHead(Q, K, V)
    z = attention_head(x)
    assert z.shape == (2, 10)

def test_multi_attention():
    Q, K, V, heads = 10, 10, 10, 5
    x = torch.randn(2, 10)
    multi_attention = MultiAttention(Q, K, V, heads)
    z = multi_attention(x)
    assert z.shape == (2, 10)