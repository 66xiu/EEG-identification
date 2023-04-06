# 最大池化maxpool
最大池化是要保留输入的特征，同时要将数据量减少
```
import torch
from torch import nn
from torch.nn import MaxPool2d

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]],dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


m = Module()
output = m(input)
print(output)
```
