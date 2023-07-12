import torch
from models.vanilla_transformers import ( 
                   EncoderBlock, 
                   FFN, 
                   Encoder, 
                   MultiHeadAttention,
                   PositionalEncoding)
from models.BERT import BERT

def test_ffn():
    input_size, hidden_size, output_size, num_layers, ffn_pdrop = 10, 20, 10, 4, 0.2
    x = torch.randn(2, 10)
    ffn = FFN(input_size, hidden_size, output_size, num_layers, ffn_pdrop)
    z = ffn(x)
    assert z.shape == (2, 10)
    
def test_multi_head_attention():
    x = torch.randn(3, 2, 200)
    emb_dim, num_attention_heads, attn_pdrop, linear_proj_pdrop = 200, 2, 0.2, 0.2
    multi_attention = MultiHeadAttention(emb_dim=emb_dim,
                                         num_attention_heads=num_attention_heads,
                                         attn_pdrop=attn_pdrop,
                                         linear_proj_pdrop=linear_proj_pdrop)
    z = multi_attention(x)
    assert z.shape == (3, 2, 200)

def test_encoder_block():
    emb_dim, num_attention_heads = 200, 2
    hidden_size, num_layers, ffn_pdrop = 20, 4, 0.2
    attn_pdrop, linear_proj_pdrop = 0.2, 0.2
    x = torch.randn(3, 2, 200)
    single_encoder_block = EncoderBlock(emb_dim=emb_dim, num_attention_heads=num_attention_heads, 
                                   hidden_size=hidden_size, num_layers=num_layers, 
                                   ffn_pdrop=ffn_pdrop, attn_pdrop=attn_pdrop, 
                                   linear_proj_pdrop=linear_proj_pdrop)
    z = single_encoder_block(x)
    assert z.shape == (3, 2, 200)

def test_encoder():
    emb_dim, num_attention_heads = 200, 2
    hidden_size, num_layers, ffn_pdrop = 20, 4, 0.2
    attn_pdrop, linear_proj_pdrop = 0.2, 0.2
    pos_emb_pdrop, emb_pdrop = 0.2, 0.2
    vocab_size, max_seq_len, num_encoder_blocks = 50000, 512, 3

    x = torch.randint(0, vocab_size, (3, 2)) #create 3 sequences of length 2, i.e 3 batches of 2 words
    multi_encoder = Encoder(emb_dim=emb_dim, num_attention_heads=num_attention_heads,
                            hidden_size=hidden_size, num_layers=num_layers,
                            ffn_pdrop=ffn_pdrop, attn_pdrop=attn_pdrop,
                            linear_proj_pdrop=linear_proj_pdrop,
                            pos_emb_pdrop= pos_emb_pdrop, emb_pdrop=emb_pdrop,
                            vocab_size=vocab_size, max_seq_len=max_seq_len, 
                            num_encoder_blocks=num_encoder_blocks)
    z = multi_encoder(x)
    assert z.shape == (3, 2, 200)

def test_postional_encoding():
    pos_emb_pdrop, max_seq_len = 0.2, 512
    x = torch.randn(3, 2, 10)
    pos_encoding = PositionalEncoding(10, pos_emb_pdrop, max_seq_len)
    z = pos_encoding(x)
    assert z.shape == (3, 2, 10)