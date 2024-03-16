import torch
import numpy as np
class NumpyToTorchTensor:
    def __init__(self, dtype):
        self.dtype = dtype
    def __call__(self, sample ):
        return torch.tensor(sample.tolist(),dtype=self.dtype)