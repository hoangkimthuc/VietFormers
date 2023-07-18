from utils import getMask
import os
from torch import nn

def test_getMask():
    mask = getMask(seq_len=4)
    assert mask.shape == (4, 4)