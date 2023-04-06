# 基本骨架 nn.Module 使用

```
import torch
from torch import nn


class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, index):
        output = index + 1
        return output


m = Module()
x = torch.tensor(1.0)
output = m(x)

print(output)
```



