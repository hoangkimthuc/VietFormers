from utils import getPositionEncoding, getMask

def test_getPositionEncoding():
    P = getPositionEncoding(seq_len=4, d=40, n=100)
    assert P.shape == (4, 40)

def test_getMask():
    mask = getMask(seq_len=4)
    assert mask.shape == (4, 4)