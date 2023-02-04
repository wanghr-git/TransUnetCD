import torch
import torch.nn as nn

class SpatialMaxPool(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SpatialMaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        
    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.shape[0], -1, 1, 1)
        return x


if __name__ == "__main__":
    input_1 = torch.rand((8, 3, 512, 512))
    pool = SpatialMaxPool(2)
    test = pool(input_1)
    print('111')