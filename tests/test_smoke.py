import torch
from esa import ESA
def test_smoke():
    x=torch.randn(2,5,16); assert ESA(16,backend="pytorch")(x).shape==x.shape
