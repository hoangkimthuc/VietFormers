from models.vanilla_transformers import Encoder
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

class MLMHead(nn.Module):
    def __init__(self, emb_dim:int, vocab_size:int, sq_len:int):
        super(MLMHead, self).__init__()
        self.linear = nn.Linear(emb_dim*sq_len, vocab_size)
        
    def forward(self, x: Tensor):
        """
        x: (batch_size, sequence_length, emb_dim)
        """
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x
    
class BERT(nn.Module):
    def __init__(self, emb_dim:int, num_attention_heads:int, 
                 hidden_size:int, num_layers:int, ffn_pdrop:float, 
                 attn_pdrop:float, linear_proj_pdrop:float, 
                 pos_emb_pdrop:float, emb_pdrop:float, bert_pdrop:float,
                 vocab_size:int, max_seq_len:int, sq_len:int, num_encoder_blocks:int):
        super(BERT, self).__init__()
        self.encoder = Encoder(emb_dim=emb_dim, num_attention_heads=num_attention_heads,
                            hidden_size=hidden_size, num_layers=num_layers,
                            ffn_pdrop=ffn_pdrop, attn_pdrop=attn_pdrop,
                            linear_proj_pdrop=linear_proj_pdrop,
                            pos_emb_pdrop= pos_emb_pdrop, emb_pdrop=emb_pdrop,
                            vocab_size=vocab_size, max_seq_len=max_seq_len, 
                            num_encoder_blocks=num_encoder_blocks)
        self.mlm_head = MLMHead(emb_dim=emb_dim, vocab_size=vocab_size, sq_len=sq_len)
        self.bert_pdrop = nn.Dropout(bert_pdrop)
    
    def forward(self, x: Tensor):
        """
        x: (batch_size, sequence_length)
        """
        x = self.encoder(x)
        x = self.mlm_head(x) 
        x = self.bert_pdrop(x)
        return x
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, seq_len, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        
        while idx.size()[-1] < max_new_tokens:
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= seq_len else idx[:, -seq_len:]
            # forward the model to get the logits for the index in the sequence
            if idx_cond.size()[-1] != seq_len:
                break
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits/ temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx