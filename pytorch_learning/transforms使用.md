# transforms使用

```
from torchvision import transforms
```

![](D:\Program Files\Typore_image\微信图片_20230404120056.png)



## 图片格式--打开方式

​    PIL---------->Image.open()

​    tensor----->ToTensor()

​    narrays---->cv.imread()



```
img_path = r"hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()  //将PIL Image 或 numpy.ndarray 转为tensor
tensor_img1 = tensor_trans(img)    //PIL to tensor
print(tensor_img1)

import cv2
cv_img = cv2.imread(img_path)
tensor_img2 = tensor_trans(cv_img)
```

--------

## transforms方法

### TOTensor

```
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
```

### Normalize--归一化

```
trans_norm = transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
```

### resize

```
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)  //img PIL---> resize----> img_resize  PIL
img_resize = trans_totensor(img_resize)   //img_resize  PIL--->totensor---> img_resize tensor
```

```
trans_resize_2 = transforms.Resize（512）
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])  //PIL -> PIL-> tensor
img_resize_2 = trans_compose(img)
```

### RandomCrop 随机裁剪

```
trans_random = transforms.RandomCrop(512)  //trans_random = transforms.RandomCrop(500,100)
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
		img_crop = trans_compose_2(img)
		write.add_image("RandomCrop",img_crop,i)
```







