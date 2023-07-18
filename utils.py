import torch 
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
def getMask(seq_len):
    mask = torch.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            if i <= j:
                mask[i, j] = 1
    return mask

def draw_histogram(tensor: torch.Tensor, title: str, epoch = int, bins: int = 50):    
    tensor_to_array = tensor.detach().cpu().numpy().flatten()            
    plt.hist(tensor_to_array, bins=bins)
    plt.title(title)
    plt.savefig(Path("visualizations/"+"epoch_"+str(epoch)+"_"+title+".png"))