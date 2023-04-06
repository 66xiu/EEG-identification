# nn.conv2d写卷积层
```
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, 64, )


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


m = Module()
print(m)

writer = SummaryWriter("./logs")
step = 0
for data in dataloader:
    imgs, target = data
    output = m(imgs)
    print(imgs)
    print(output)
    # torch.Size([64,3,32,32])
    writer.add_images("inputs", imgs, step)
    # torch.Size([64,6,30,30])-->[xxx,3,30,30]
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("outputs", output, step)

    step = step + 1

writer.close()

```
