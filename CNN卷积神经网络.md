# CNN卷积神经网络

## 1.卷积层

卷积层主要就是通过不同的卷积核来提取图片中不同的特征信息，例如横向的信息，纵向的信息之类的。通过多层卷积，对于人眼而言，特征逐渐变得抽象，但信息也变得更加丰富，从边缘到纹理，再到更深层次。

<img src="file:///C:\Users\s'j'y\Documents\Tencent Files\2841247687\Image\C2C\5F62FE891435E7C695FDC2F1AD6A4DC9.png" alt="img" style="zoom:50%;" />

## 2.池化层

池化层的作用主要是降低计算量和降低优化难度。原始大小的特征图直接解码成类别难度较高，通过池化层降低特征图大小，使得优化难度降低。

## 3.非线性层

非线性层的主要作用是加强特征，例如经典的ReLU函数，小于零的部分统统置为0，而保留0以上的特征，为神经网络添加了非线性的部分，更加符合现实生活场景

## 4.全连接层

全连接层实际上主要起到了分类器的作用。假设我要进行一个图片分类的工作，最后通过两个线性层，我得到了100个神经元，也就是100个特征，经过训练之后，在模型中激活其中特定的59个神经元的图片的target为猫，那么我将一张图片通过该模型识别后，它能够激活那59个神经元中的58个，那么就很有理由认为这张图片上面的事物是个猫。

<img src="file:///C:\Users\s'j'y\Documents\Tencent Files\2841247687\Image\C2C\6EBEC3C7DCDDF0E4E0C4716C51341091.jpg" alt="img" style="zoom:67%;" />

[【卷积神经网络可视化】 从卷积层到池化层的可视化演示（中英双语字幕）-哔哩哔哩](https://www.bilibili.com/video/BV1nU4y187sX?p=1&share_medium=android&share_plat=android&share_session_id=adcf5af0-b66e-4a6a-8c72-afb092295223&share_source=COPY&share_tag=s_i&timestamp=1657447279&unique_k=BYAiljv) 