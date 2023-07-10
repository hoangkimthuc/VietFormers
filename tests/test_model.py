import torch
from models.vanilla_transformers import ( 
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

# def test_attention_head():
#     emb_dim = 10
#     x = torch.randn(3, 2, 10) #batch size, sequence len, embedding size
#     attention_head = AttentionHead(emb_dim)
#     z = attention_head(x)
#     assert z.shape == (3, 2, 10)
    
def test_multi_head_attention():
    x = torch.randn(3, 2, 200)
    multi_attention = MultiHeadAttention()
    z = multi_attention(x)
    assert z.shape == (3, 2, 200)

def test_encoder_block():
    emb_dim, num_attention_heads = 200, 2
    x = torch.randn(3, 2, 200)
    multi_attention = EncoderBlock(emb_dim, num_attention_heads)
    z = multi_attention(x)
    assert z.shape == (3, 2, 200)

def test_encoder():
    emb_dim, num_attention_heads, num_encoder_blocks = 10, 5, 3
    x = torch.randint(0, 50000, (3, 2)) #create 3 sequences of length 2, i.e 3 batches of 2 words
    multi_encoder = Encoder(emb_dim=emb_dim, vocab_size=50000, num_attention_heads=num_attention_heads, num_encoder_blocks=num_encoder_blocks)
    z = multi_encoder(x)
    assert z.shape == (3, 2, 10)

def test_postional_encoding():
    x = torch.randn(3, 2, 10)
    pos_encoding = PositionalEncoding(10)
    z = pos_encoding(x)
    assert z.shape == (3, 2, 10)

def test_bert():
    emb_dim, num_attention_heads, num_encoder_blocks = 10, 5, 3
    x = torch.randint(0, 50000, (3, 2)) #create 3 sequences of length 2, i.e 3 batches of 2 words
    bert = BERT(emb_dim=emb_dim, vocab_size=50000, num_attention_heads=num_attention_heads, num_encoder_blocks=num_encoder_blocks)
    z = bert(x)
    assert z.shape == (3, 2, 50000)