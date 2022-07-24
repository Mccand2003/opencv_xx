# Tensorboard使用

## 1、SummaryWriter()

这个函数用于创建一个tensorboard文件

常用参数：

log_dir：tensorboard文件的存放路径

flush_secs：表示写入tensorboard文件的时间间隔

例：

```python
writer = SummaryWriter("logs")
```

运行后会在当前位置生成一个文件夹，里面存储着event文件，需要根据这

个文件进行可视化

## 2、writer.add_graph()

这个函数用于在tensorboard中创建Graphs，Graphs中存放了网络结构，

对神经网络结构进行可视化

常用参数：

model：pytorch模型

input_to_model：pytorch模型的输入

例：

```python
writer.add_graph(model, (graph_inputs,))
```

## 3、writer.add_scalar()

这个函数用于在tensorboard对损失进行可视化，也可以对准

确率之类的进行可视化

常用参数：

tag：坐标系名称

scalar_value：具体的值，即y轴数据

global_step ：x轴坐标

例：

```python
writer.add_scalar("test_loss", total_test_loss, total_test_step)
```

![image-20220724153347654](C:\Users\s'j'y\AppData\Roaming\Typora\typora-user-images\image-20220724153347654.png)

## 4、writer.add_images

绘制图片，可用于检查模型的输入，监测特征图的变化，或是观察图片效

果 

tag：就是保存图的名称

img_tensor:图片的类型要是torch.Tensor, numpy.array, or string这三种

global_step：第几张图片

dataformats=‘CHW’，默认CHW，tensor是CHW，以opencv读取的图片

要HWC

例：

```python
writer.add_images("input", imgs, step)
```

![image-20220724153731161](C:\Users\s'j'y\AppData\Roaming\Typora\typora-user-images\image-20220724153731161.png)

## 5、writer.close

关闭窗口

## 6、使用方法

 tensorboard --logdir 'logs'

单引号内为存放文件的文件夹名称，按下回车键后会出现网址，点击进入即可

![image-20220724153751778](C:\Users\s'j'y\AppData\Roaming\Typora\typora-user-images\image-20220724153751778.png)

tensorboard --logdir ‘logs’  --port=6007

也可以通过在后面增加指令，改变端口

