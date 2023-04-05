# tensorBoard使用

```
from torch.utils.tensorboard import SummaryWriter
```

## 创建对象

```
writer = SummaryWriter("logs")  //放在logs文件夹里
```

## 方法add_scalar()

```
for i in range(100):
	writer.add_scalar("x=y",i,i)
	
writer.close()
```

## 查看

```
tensorboard --logdir=logs
```

## 将PIL..类型转为numpy类型

```
from PIL import Image
import numpy as np

img = r"D:\桌面\pytorch_learning\hymenoptera_data\train\ants\0013035.jpg"
img = Image.open(image_path)  //img是PIL..类型
img_array = np.array(img)   //img_array是numpy类型

```

## 方法add_image()

```
writer.add_image("test",img_array,1,dataformats='HWC')
```



