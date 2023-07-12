from models.vanilla_transformers import Encoder
from torch import nn
from torch import Tensor

class BERT(nn.Module):
    def __init__(self, emb_dim:int, num_attention_heads:int, 
                 hidden_size:int, num_layers:int, ffn_pdrop:float, 
                 attn_pdrop:float, linear_proj_pdrop:float, 
                 pos_emb_pdrop:float, emb_pdrop:float, bert_pdrop:float,
                 vocab_size:int, max_seq_len:int, num_encoder_blocks:int):
        super(BERT, self).__init__()
        self.encoder = Encoder(emb_dim=emb_dim, num_attention_heads=num_attention_heads,
                            hidden_size=hidden_size, num_layers=num_layers,
                            ffn_pdrop=ffn_pdrop, attn_pdrop=attn_pdrop,
                            linear_proj_pdrop=linear_proj_pdrop,
                            pos_emb_pdrop= pos_emb_pdrop, emb_pdrop=emb_pdrop,
                            vocab_size=vocab_size, max_seq_len=max_seq_len, 
                            num_encoder_blocks=num_encoder_blocks)
        self.linear = nn.Linear(emb_dim, vocab_size)
        self.bert_pdrop = nn.Dropout(bert_pdrop)
    
    def forward(self, x: Tensor):
        """
        x: (batch_size, sequence_length)
        """
        x = self.encoder(x)
        x = self.linear(x) 
        x = self.bert_pdrop(x)
        return x