# datasets
```dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])```

```
  train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
  test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
```
      
----------
# dataloader

```
  test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor())
  test_load = DataLoader(dataset=test_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

  for data in test_load:
      imgs, targets = data
    
```
