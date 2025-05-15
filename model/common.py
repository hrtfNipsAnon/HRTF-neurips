import torch.nn as nn

initial_size_to_strides_map = {
    864: [2, 2, 2, 2],
    100: [2, 2, 2, 2],
    27: [2, 2, 2, 1],
    25: [2, 2, 2, 1],
    18: [2, 2, 1, 1],
    16: [2, 2, 1, 1],
    9: [2, 1, 1, 1],
    8: [2, 1, 1, 1],
    5: [1, 1, 1, 1],
    4: [1, 1, 1, 1],
    3: [1, 1, 1, 1]
}

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    
    def forward(self, x):
        return x.view(self.shape)

class Trim(nn.Module):
    def __init__(self, shape, dim=-1):
        super().__init__()
        self.shape = shape
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return x[:,:,:self.shape]
        return x[:,:self.shape,...]