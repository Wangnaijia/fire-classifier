from torchvision.models.resnet import resnet18
import os,time
import random
from PIL import Image
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
 
 
class Fire(data.Dataset):
    def __init__(self, root, transform=None, train=True, test=False):
        self.test = test
        self.train = train
        self.transform = transform
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
 
        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg
 
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-1].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-1]))
        imgs_num = len(imgs)
        if self.test:
            self.imgs = imgs
        else:
            random.shuffle(imgs)
            if self.train:
                self.imgs = imgs[:int(0.7 * imgs_num)]
            else:
                self.imgs = imgs[int(0.7 * imgs_num):]
 
    # 作为迭代器必须有的方法
    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0  # 狗的label设为1，猫的设为0
        data = Image.open(img_path)
        data = self.transform(data)
        return data, label
 
    def __len__(self):
        return len(self.imgs)
 

# 对数据集训练集的处理
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),  # 先调整图片大小至256x256
    transforms.RandomCrop((224, 224)),  # 再随机裁剪到224x224
    transforms.RandomHorizontalFlip(),  # 随机的图像水平翻转，通俗讲就是图像的左右对调
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))  # 归一化，数值是用ImageNet给出的数值
])
 
# 对数据集验证集的处理
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 若能使用cuda，则使用cuda
# trainset = DogCat('Train_CT', transform=transform_train)
# valset = DogCat('test', transform=transform_val)
train_dataset = torchvision.datasets.ImageFolder(root='train',transform=transform_train)
trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=20,shuffle=True)
# hard_example = torchvision.datasets.ImageFolder(root='hard_example',transform=transform_train)
# hard_example_loader = torch.utils.data.DataLoader(hard_example, batch_size=20,shuffle=True)
# print(hard_example.classes)
# print(hard_example.class_to_idx)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=0)
# valloader = torch.utils.data.DataLoader(valset, batch_size=20, shuffle=False, num_workers=0)
 
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total
 
 
def train(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    model.train()
    train_acc = 0.0
    train_loss=0
    train_accuracy=0
    t1 = time.time()
    for batch_idx, (img, label) in enumerate(trainloader):
        image = Variable(img.cuda())
        label = Variable(label.cuda())
        optimizer.zero_grad()
        out = model(image)
        # print('out:{}'.format(out))
        # print(out.shape)
        # print('label:{}'.format(label))
        loss = criterion(out, label)
        train_loss += loss
        loss.backward()
        optimizer.step()
        train_acc = get_acc(out, label)
        train_accuracy += train_acc
        
        print("Epoch:%d [%d|%d] loss:%f acc:%f" % (epoch, batch_idx, len(trainloader), loss.mean(), train_acc))
    t2 = time.time()
    writer.add_scalar("Train_adam/loss",train_loss/len(trainloader),epoch)
    writer.add_scalar("Train_adam/acc",100.0 *train_accuracy/len(trainloader),epoch)
    f = open('model_result/{}_adam_with_hard.txt'.format(modelname), 'a+')
    f.write("Epoch:%d  loss:%f acc:%f time cost:%f" % (epoch, train_loss/len(trainloader), 100.0 *train_accuracy/len(trainloader), t2-t1))
    f.write('\n')
    f.close()
 
 
def val(epoch):
    print("\nValidation Epoch: %d" % epoch)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(valloader):
            image = Variable(img.cuda())
            label = Variable(label.cuda())
            out = model(image)
 
            _, predicted = torch.max(out.data, 1)
 
            total += image.size(0)
            correct += predicted.data.eq(label.data).cpu().sum()
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))
    writer.add_scalar("Val/acc",((1.0 * correct.numpy()) / total),epoch)
 
 
class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # 取掉model的后1层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(512, 2) #加上一层参数修改好的全连接层
 
    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x
 
if __name__ =='__main__':
    resnet = resnet18(pretrained=True)
    # model = Net(resnet)
    model = torch.load('Resnet18_fire.pth')  # 加载模型
    model = model.to(device)
    # model.summary()
    modelname = "Resnet18"
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)  # 设置训练细节
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=3)
    criterion = nn.CrossEntropyLoss()
    # 定义tensorboard运行位置
    writer = SummaryWriter("runs/{}_new_fire1".format(modelname)) # 加入难例的
    for epoch in range(30):
        train(epoch)
        # val(epoch)
    print("saving model...")
    torch.save(model, modelname+"_fire_with_hard.pth")  # 保存模型