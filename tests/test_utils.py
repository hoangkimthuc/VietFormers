from utils import getMask

def test_getMask():
    mask = getMask(seq_len=4)
    assert mask.shape == (4, 4)