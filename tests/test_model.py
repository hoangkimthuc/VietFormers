import torch
from models.vanilla_transformers import (AttentionHead, 
                   EncoderBlock, 
                   FFN, 
                   Encoder, 
                   MultiHeadAttention,
                   PositionalEncoding)
from models.BERT import BERT
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
    x = torch.randint(0, 50000, (3, 2)) #create 3 sequences of length 2, i.e 3 batches of 2 words
    multi_encoder = Encoder(Q_dim=Q, K_dim=K, V_dim=V, vocab_size=50000, num_attention_heads=heads, num_encoder_blocks=num_encoder, max_seq_len=2)
    z = multi_encoder(x)
    assert z.shape == (3, 2, 10)

def test_postional_encoding():
    x = torch.randn(3, 2, 10)
    pos_encoding = PositionalEncoding(10)
    z = pos_encoding(x)
    assert z.shape == (3, 2, 10)

def test_bert():
    Q, K, V, heads, num_encoder = 10, 10, 10, 5, 3
    x = torch.randint(0, 50000, (3, 2)) #create 3 sequences of length 2, i.e 3 batches of 2 words
    bert = BERT(Q_dim=Q, K_dim=K, V_dim=V, vocab_size=50000, num_attention_heads=heads, num_encoder_blocks=num_encoder, max_seq_len=2)
    z = bert(x)
    assert z.shape == (3, 2, 50000)