# 线性层Linear
https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
```
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64,drop_last=True)


class module(nn.Module):
    def __init__(self):
        super(module, self).__init__()
        self.Linear1 = Linear(in_features=196608, out_features=10)

    def forward(self, input):
        output = self.Linear1(input)
        return output


m = module()

for data in dataloader:
    imgs, target = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = m(output)
    print(output.shape)

```
