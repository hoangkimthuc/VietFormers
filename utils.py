import torch 

def getMask(seq_len):
    mask = torch.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            if i <= j:
                mask[i, j] = 1
    return mask
