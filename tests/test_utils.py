from utils import getPositionEncoding

def test_getPositionEncoding():
    P = getPositionEncoding(seq_len=4, d=40, n=100)
    assert P.shape == (4, 40)
