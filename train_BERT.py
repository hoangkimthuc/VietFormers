from models.BERT import BERT
import math
from typing import Tuple, Union
import hydra
import wandb
import datetime

import torch
from torch import nn, Tensor
from torch.utils.data import dataset
from torch.utils.data.dataset import IterableDataset
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from einops import rearrange
from utils import draw_histogram

import time
from dotenv import load_dotenv
load_dotenv()
# *** Train BERT on wikitext-2 dataset ***

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_process(raw_text_iter: IterableDataset, vocab, tokenizer) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

def get_batch(source: Tensor, i: int, sq_len: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    sq_len = min(sq_len, len(source) - 1 - i)
    data_and_target = rearrange(source[i:i+sq_len+1], 'seq batch -> batch seq')
    data = data_and_target[: , :sq_len]
    #Target is the next word of each sequence in the batch
    target = data_and_target[:, -1]
    return data, target

def train(model: nn.Module, train_data: Tensor, val_data: Tensor, 
          num_epochs:int, criterion, lr:Union[float, int], 
          optimizer, scheduler, ntokens:int, sq_len:int,
          log_interval: int, wandb_enabled: bool,
          vocab, tokenizer) -> None:
    count = 0
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):       
        epoch_start_time = time.time()
        model.train()  # turn on train mode
        total_loss = 0.
        log_interval = log_interval
        start_time = time.time()

        num_batches = len(train_data) // sq_len
        for batch, i in enumerate(range(0, train_data.size(0) - 1, sq_len)):
            data, targets = get_batch(train_data, i, sq_len)
            if data.shape[1] != sq_len:
                continue
            output = model(data)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()
                if wandb_enabled: 
                    wandb.log({"loss": cur_loss}, step = batch + (epoch-1)*num_batches)           
        
        context = '''The government is an organization which provides rules and protection to the people it governs.
        Also, it is a system of ruling the people. The government has a legislature, executive and a judiciary.
        The government is a body that has the power to make and enforce laws for a country or community.
        The government has the power to make laws, to enforce them and to settle disputes about them.'''
        x = torch.tensor(vocab(tokenizer(context)))[None,...].to(device)
        y = model.generate(x, 200, seq_len=sq_len, top_k=10, do_sample=True).squeeze().tolist()
        completion = ' '.join(vocab.lookup_tokens(y))
        print(completion)
        
        val_loss = evaluate(model, val_data,criterion,sq_len,ntokens)
        val_ppl = math.exp(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), "best_model_params.pt")
        else:
            count += 1
        if count == 5:
            print("Early stopping")
            break        
        if wandb_enabled:
            wandb.log({"val_loss": val_loss, "epoch": epoch})
            wandb.log({"val_ppl": val_ppl, "epoch": epoch})

        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)
        scheduler.step()
    
def evaluate(model: nn.Module, eval_data: Tensor, criterion, sq_len:int, ntokens:int,) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, sq_len):
            data, targets = get_batch(eval_data, i, sq_len)
            if data.shape[1] != sq_len:
                continue
            seq_len = data.size(0)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len*criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


@hydra.main(version_base=None, config_path="conf", config_name="BERT")
def main(cfg) -> None:
    # Prepare data
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    # ``train_iter`` was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter, vocab, tokenizer)
    val_data = data_process(val_iter, vocab, tokenizer)
    test_data = data_process(test_iter, vocab, tokenizer)
    
    ntokens = len(vocab)  # size of vocabulary
    train_data = batchify(train_data, cfg.training.train_batch_size)  # shape ``[seq_len, batch_size]``
    val_data = batchify(val_data, cfg.training.eval_batch_size)
    test_data = batchify(test_data, cfg.training.eval_batch_size)
    
    
    model = BERT(emb_dim=cfg.model.emb_dim, 
                num_attention_heads=cfg.model.num_attention_heads,
                hidden_size=cfg.model.hidden_size, 
                num_layers=cfg.model.num_layers,
                ffn_pdrop=cfg.model.ffn_pdrop, 
                attn_pdrop=cfg.model.attn_pdrop,
                linear_proj_pdrop=cfg.model.linear_proj_pdrop,
                bert_pdrop=cfg.model.bert_pdrop,
                pos_emb_pdrop= cfg.model.pos_emb_pdrop,
                emb_pdrop=cfg.model.emb_pdrop,
                vocab_size=ntokens, 
                max_seq_len=cfg.model.max_seq_len,
                num_encoder_blocks=cfg.model.num_encoder_blocks,
                sq_len=cfg.training.sq_len,)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

     # The number of epochs
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.99)

    #Add wandb for logging training loss and validation loss
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if cfg.wandb.is_enabled:
        wandb.init(
        project=cfg.wandb.project, 
        name=f"{cfg.wandb.run_name}_{current_datetime}", 
        )
        wandb.define_metric("epoch")
        wandb.define_metric("val_loss", step_metric="epoch")
        wandb.define_metric("val_ppl", step_metric="epoch")
        if cfg.wandb.watch_model:
            wandb.watch(model, log="all", 
                        criterion=criterion, 
                        log_freq=cfg.wandb.log_freq, 
                        log_graph=True)
    
    train(model=model,
          train_data=train_data,
          val_data=val_data,
          num_epochs=cfg.training.num_epochs,
          criterion=criterion,
          lr=cfg.training.lr, 
          optimizer=optimizer, 
          scheduler=scheduler,
          ntokens=ntokens, 
          sq_len=cfg.training.sq_len ,
          log_interval=cfg.training.log_interval,
          wandb_enabled=cfg.wandb.is_enabled,
          vocab=vocab,
          tokenizer=tokenizer,) 


if __name__ == "__main__":
    main()
    