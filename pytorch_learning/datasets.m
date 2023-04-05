```dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])```


  train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
  test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
           
      
