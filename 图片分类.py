import torchvision as tv
import torch as t
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
show = ToPILImage()

# 定义对数据的预处理
transform = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
                             ])
# 训练集
trainset = tv.datasets.CIFAR10(
                    root='CIFAR10_dataset',
                    train=True,
                    download=True,
                    transform=transform)

trainloader = t.utils.data.DataLoader(
                    trainset,
                    batch_size=4,
                    shuffle=True,
                    num_workers=2)

# 测试集
testset = tv.datasets.CIFAR10(
                    'CIFAR10_dataset',
                    train=False,
                    download=True,
                    transform=transform)

testloader = t.utils.data.DataLoader(
                    testset,
                    batch_size=4,
                    shuffle=False,
                    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

(data, label) = trainset[100]
# print(classes[label])


show((data + 1) / 2).resize((100, 100))
if __name__ == '__main__':

    dataiter = iter(trainloader)
    images, labels = dataiter.next()  # 返回4张图片及标签
    print(' '.join('%11s' % classes[labels[j]] for j in range(4)))
    show(tv.utils.make_grid((images+1)/2)).resize((400, 100))

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16*5*5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(x.size()[0], -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()
    print(net)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    t.set_num_threads(8)
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # 输入数据
            inputs, labels = data
            inputs, labels = inputs, labels
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 更新参数
            optimizer.step()

            # 打印log信息
            running_loss += loss.item()
            if i % 2000 == 1999:  # 每2000个batch打印一下训练状态
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    dataiter = iter(testloader)
    images, labels = dataiter.next()  # 一个batch返回4张图片
    print('实际的label: ', ' '.join(
                '%08s' % classes[labels[j]] for j in range(4)))
    show(tv.utils.make_grid(images / 2 - 0.5)).resize((400, 100))
    # 计算图片在每个类别上的分数
    outputs = net(images)
    # 得分最高的那个类
    _, predicted = t.max(outputs.data, 1)
    print('预测结果: ', ' '.join('%5s'\
                % classes[predicted[j]] for j in range(4)))
    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = t.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))