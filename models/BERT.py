from models.vanilla_transformers import Encoder
from torch import nn
from torch import Tensor

class BERT(nn.Module):
    def __init__(self, emb_dim:int, vocab_size:int, max_seq_len:int=512, num_attention_heads:int=5, num_encoder_blocks:int=3, hidden_size:int=512, num_layers:int=4, dropout_p:float=0.1):
        super(BERT, self).__init__()
        self.encoder = Encoder(emb_dim=emb_dim,  
                               vocab_size=vocab_size,
                               max_seq_len=max_seq_len,
                               num_attention_heads=num_attention_heads, 
                               num_encoder_blocks=num_encoder_blocks,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout_p=dropout_p)

        self.linear = nn.Linear(emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x: Tensor):
        """
        x: (batch_size, sequence_length)
        """   
        x = self.encoder(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x