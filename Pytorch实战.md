# Pytorch实战

## 1.transforms的使用

```python
from torchvision import transforms
```

### 1.1 图像加载

可以使用Opencv或Image.open()

```python
img = cv2.imread("C:\\Users\\s'j'y\Desktop\dataset\\train\\ants_image\\460874319_0a45ab4d05.jpg")
pil = Image.open("C:\\Users\\s'j'y\Desktop\dataset\\train\\ants_image\\460874319_0a45ab4d05.jpg")
```

采用绝对路径，将\改为双斜杠（\为转义字符，故双斜杠表示\）

### 1.2 常用函数

#### 1.2.1  ToTensor

```python
def __call__(self, pic):
    """
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    return F.to_tensor(pic)
```

如代码，将图片转化为tensor类型，灰度范围转化为0到1，shape为（C，H，W），tensor类型具有特殊的属性，梯度和反向传播

注：opencv读取图片为nump.ndarray，其shape为（H，W，C），灰度范围0到255

<img src="C:\Users\s'j'y\AppData\Roaming\Typora\typora-user-images\image-20220708213218521.png" alt="image-20220708213218521" style="zoom:55%;" />

使用范例：

```python
trans_totensor = transforms.ToTensor()
img_tenser = trans_totensor(img)
```

#### 1.2.2  Normalize

```python
def __init__(self, mean, std, inplace=False):
    super().__init__()
    _log_api_usage_once(self)
    self.mean = mean
    self.std = std
    self.inplace = inplace
```

mean：均值

std：标准差

image=(image-mean)/std

通常将[0,1]映射成[-1,1]的标准正态分布，方便模型训练，mean和std应根据数据集进行计算

使用范例：

```python
trans_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_norm = trans_norm(img_tenser)
```

#### 1.2.3 Resize

```python
def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
```

size：1.只设置一个参数x，则等比缩放（短边为x）

​           2.设置参数（H，W）

对于PIL Image和tensor类型进行resize的运算

使用范例：

```python
trans_resize = transforms.Resize((200, 300))
img_resize = trans_resize(img_tenser)
```

#### 1.2.4 Compose

```python
def __init__(self, transforms):
```

transforms：一个list，list里是多个'Transform'对象，即[transforms, transforms...]。

用来组合多个torchvision.transforms操作，遍历list，对img依次执行每个transforms操作，并返回transforms后的img。

使用范例：

```
trans_resize_2 = transforms.Resize(300)
# PIL -> PIL Resise ->tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(pil)
```

#### 1.2.5 RandomCrop

```python
def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
```

size：裁剪后尺寸，（H，W）或W

对tensor类型，PIL类型进行随机裁剪，进行数据扩充，提高模型精度，

增强模型稳定性

使用范例：

```python
trans_random = transforms.RandomCrop((200,50))
```

ps.还有CenterCrop（中心裁剪），RandomResizedCrop（随机缩放裁剪）等

# 2 数据加载

## 2.1 数据集

### 2.1.1 使用torchvison中的数据集

```python
import torchvision
```

#### 2.1.1.1 DataSet

以CIFA10数据集为例

```python
def __init__(
    self,
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
)
```

root：数据集存储路径

train：True为训练集，False为测试集

transform：图像处理，使用transforms中的函数

download：True则在root路径中下载数据集，False则不下载（ps.下载过慢，可通过迅雷下载好后，放入root路径中）

返回值：  img， target = ttrain_set[0]

该数据集有classes属性，target表示对应classes中的第几个

使用范例：

```python
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root="D:\python_pytorch\CIFAR10_dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="D:\python_pytorch\CIFAR10_dataset", train=False, transform=dataset_transform, download=True)
```

#### 2.1.1.2 DataLoader

```python
from torch.utils.data import DataLoader
```

```python
def __init__(self, dataset: Dataset[T_co],
             batch_size: Optional[int] = 1,
             shuffle: bool = False, 
             sampler: Union[Sampler, Iterable, None] = None,
             batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
             num_workers: int = 0,
             collate_fn: Optional[_collate_fn_t] = None,
             pin_memory: bool = False, 
             drop_last: bool = False,
             timeout: float = 0,
             worker_init_fn: Optional[_worker_init_fn_t] = None,
             multiprocessing_context=None, generator=None,
             *, prefetch_factor: int = 2,
             persistent_workers: bool = False):
```

dataset：数据集

batch_size：每组训练的数据数目

shuffle：数据集是否打乱，Ture为打乱

num_workers：加载进程数

drop_last：当数据集按batch_size划分，多余的数据是否舍去（例如数据集有200图片，batch_size为99，则多余的2张是否舍去）

使用范例：

```python
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
```

经过batch_size打包后：

torch.Size([64, 3, 32, 32])
tensor([3, 2, 8, 3, 4, 6, 0, 1, 9, 5, 8, 3, 8, 9, 4, 5, 7, 5, 4, 9, 7, 8, 6, 3,
        0, 2, 6, 9, 7, 1, 4, 3, 8, 9, 5, 1, 5, 0, 6, 2, 4, 5, 5, 4, 3, 8, 6, 4,
        3, 8, 3, 8, 8, 2, 4, 6, 0, 1, 2, 7, 7, 4, 2, 2])

```python
for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)
```

# 3. 神经网络骨架（nn.Module）

## 3.1 卷积层

卷积层通过卷积核对图片进行特征提取

3.1.1 conv2d

`torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)`

in_channels：输入通道数

out_channels：输出通道数

kernel_size：卷积核大小

stride：卷积核移动步长

padding：在矩阵周围补0，数字决定补几圈

```
class Mcc(nn.Module):
    def __init__(self):
        super(Mcc, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x
```

## 3.2 池化层

池化层对特征进一步进行提取，压缩了特征大小，减小数据大小，利于运算

### 3.2.1 MaxPools

在实践中，最大池化的效果比较好，同时也有平均池化就是把取最大值

变为取平均值

kernel_size(int or tuple) ： max pooling的窗口大小

stride(int or tuple, optional) ： max pooling的窗口移动的步长。默认

值是kernel_size

padding(int or tuple, optional) ：输入的每一条边补充0的层数

dilation(int or tuple, optional) ：一个控制窗口中元素步幅的参数

return_indices ：如果等于True，会返回输出最大值的序号，对于上采

样操作会有帮助

ceil_mode ： 如果等于True，计算输出信号大小的时候，会使用向上取

整，代替默认的向下取整的操作

例：

```python
torch.nn.MaxPool2d(2)
```

## 3.3 线性层

```python
def __init__(self, in_features: int, out_features: int, bias: bool = True,
             device=None, dtype=None) -> None:
```

in_features：输入的二维张量的大小，即输入的[batch_size, size]

中的size。

out_features：指的是输出的二维张量的大小，即输出的二维张量的

形状为[batch_size，output_size]，当然，它也代表了该全连接层的神

经元个数。

从输入输出的张量的shape角度来理解，相当于一个输入为

[batch_size, in_features]的张量变换成了[batch_size, out_features]的

输出张量。

例：

```python
...
self.fc1 = nn.Linear(16*5*5, 120)
self.fc2 = nn.Linear(120, 84)
self.fc3 = nn.Linear(84, 10)
...
```

## 3.4 非线性层

### 3.4.1 ReLU

```python
def relu(input: Tensor, inplace: bool = False) -> Tensor:
```

非线性激活，该函数将小于零的部分直接置为0，大于0的部分不变，起

到了加强特征的作用，引入了非线性，使得模型可以应用于各种非线性

情景

input：一个tensor类型数据

inplace：inplace参数如果设为True，它会把输出直接覆盖到输入中，这

样可以节省内存/显存。

```python
import torch.nn.functional as F
...
x = F.relu(self.fc1(x))
...
```

# 4.其他

## 4.1 交叉熵损失函数

从经过神经网络预测的标签和实际标签进行运算，得到loss，后续将其

送入优化器对权重和偏差进行优化

例：

```python
loss_fn = torch.nn.CrossEntropyLoss()
...
		outputs = mcc(imgs)
        loss = loss_fn(outputs, targets)
        ...
```

## 4.2 杂项

作用就是要寻找到loss最小的那个点的那个点所对应的权重和偏差，学

习率的设置，过大找到最优点的速度很慢，过小容易陷入局部最优，可

以将学习率设置为随着训练轮数的增加逐渐减小，在一定程度上能够有

比较好的效果

```python
learning_rate = 0.005
optimizer = torch.optim.SGD(mcc.parameters(), lr=learning_rate)
		...
        # 梯度清0，反向传播，优化器
		optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ...
```