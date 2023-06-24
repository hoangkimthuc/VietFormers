import torch
from model import AttentionHead, EncoderBlock, FFN, Encoder, MultiHeadAttention

def test_ffn():
    input_size, hidden_size, output_size, num_layers = 10, 20, 10, 4
    x = torch.randn(2, 10)
    ffn = FFN(input_size, hidden_size, output_size, num_layers)
    z = ffn(x)
    assert z.shape == (2, 10)

def test_attention_head():
    Q, K, V = 10, 10, 10
    x = torch.randn(3, 2, 10) #batch size, sequence len, embedding size
    attention_head = AttentionHead(Q, K, V)
    z = attention_head(x)
    assert z.shape == (3, 2, 10)
    
def test_multi_head_attention():
    Q, K, V, heads = 10, 10, 10, 5
    x = torch.randn(3, 2, 10)
    multi_attention = MultiHeadAttention(Q, K, V, heads)
    z = multi_attention(x)
    assert z.shape == (3, 2, 10)

def test_encoder_block():
    Q, K, V, num_heads = 10, 10, 10, 5
    x = torch.randn(3, 2, 10)
    multi_attention = EncoderBlock(Q, K, V, num_heads)
    z = multi_attention(x)
    assert z.shape == (3, 2, 10)

def test_encoder():
    Q, K, V, heads, num_encoder = 10, 10, 10, 5, 3
    x = torch.randn(3, 2, 10)
    multi_encoder = Encoder(Q=Q, K=K, V=V, num_attention_heads=heads, num_encoder_blocks=num_encoder, max_len=2)
    z = multi_encoder(x)
    assert z.shape == (3, 2, 10)
