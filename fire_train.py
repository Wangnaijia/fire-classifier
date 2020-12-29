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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

"""baseline方法"""
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

finetune = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 若能使用cuda，则使用cuda
# trainset = DogCat('Train_CT', transform=transform_train)
if finetune:
    valset = torchvision.datasets.ImageFolder('val_expanded', transform=transform_val)
    train_dataset = torchvision.datasets.ImageFolder(root='train_expanded',transform=transform_train)
else:
    valset = torchvision.datasets.ImageFolder('val_280', transform=transform_val)
    train_dataset = torchvision.datasets.ImageFolder(root='train_280',transform=transform_train)
trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=20,shuffle=True)
# hard_example = torchvision.datasets.ImageFolder(root='hard_example',transform=transform_train)
# hard_example_loader = torch.utils.data.DataLoader(hard_example, batch_size=20,shuffle=True)
# print(hard_example.classes)
# print(hard_example.class_to_idx)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=20, shuffle=False, num_workers=0)
 
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
    writer.add_scalar("Train/loss",train_loss/len(trainloader),epoch)
    writer.add_scalar("Train/acc",100.0 *train_accuracy/len(trainloader),epoch)
    f = open('model_result/{}.txt'.format(modelname), 'a+')
    f.write("Epoch:%d  loss:%f acc:%f time cost:%f" % (epoch, train_loss/len(trainloader), 100.0 *train_accuracy/len(trainloader), t2-t1))
    f.write('\n')
    f.close()

def val(epoch):
    print("\nValidation Epoch: %d" % epoch)
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    criteria = nn.CrossEntropyLoss()
    with torch.no_grad():
        predlist=[]
        scorelist=[]
        targetlist=[]
        for batch_idx, (img, label) in enumerate(valloader):
            image = Variable(img.cuda())
            label = Variable(label.cuda())
            out = model(image)
 
            _, predicted = torch.max(out.data, 1)
            test_loss += criteria(out.data, label.data)
            score = F.softmax(out.data, dim=1)
            total += image.size(0)
            correct += predicted.data.eq(label.data).cpu().sum()
            targetcpu = label.data.cpu().numpy()
            predlist=np.append(predlist, predicted.data.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
        print("Acc: %f " % ((1.0 * correct.numpy()) / total))
        writer.add_scalar("Test/acc",((100 * correct.numpy()) / total),epoch)
        writer.add_scalar("Test/loss",test_loss,epoch) 
    return targetlist, scorelist, predlist


if __name__ == '__main__':
    import torchvision.models as models
    # model = models.densenet121(pretrained=True).cuda()
    # modelname = 'Dense121'
    # model = models.resnet18(pretrained=True).cuda()
    # modelname = 'resnet18'
    # model = models.densenet169(pretrained=True).cuda()
    # modelname = 'Dense169'
    # model = models.resnet50(pretrained=True).cuda()
    # modelname = 'ResNet50_expanded'
    
    model = models.vgg16(pretrained=True)
    # model = model.cuda()
    # modelname = 'vgg16'
    # from efficientnet_pytorch import EfficientNet
    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    # model = model.cuda()
    # modelname = 'efficientNet-b0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelname = 'vgg16_expanded'
    model = torch.load('model_backup/vgg16_fire.pth')  # 加载模型
    model = model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)  # 设置训练细节
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=3)
    criterion = nn.CrossEntropyLoss()
    # 定义tensorboard运行位置
    writer = SummaryWriter("run_expanded/{}_on_fire".format(modelname)) # 加入难例的

    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())
    # count_pred = np.zeros(valset.__len__())

    votenum=10
    total_epoch = 10
    for epoch in range(1,total_epoch+1):
        train(epoch)
        # val(epoch)
        targetlist, scorelist, predlist = val(epoch)
        
        # count_pred[predlist <= (1/2)] = 0
        # count_pred[predlist > (1/2)] = 1
        # TP = ((count_pred == 1) & (targetlist == 1)).sum()
        # TN = ((count_pred == 0) & (targetlist == 0)).sum()
        # FN = ((count_pred == 0) & (targetlist == 1)).sum()
        # FP = ((count_pred == 1) & (targetlist == 0)).sum()
        # acc = (TP + TN) / (TP + TN + FP + FN)
        # print("test accuracy",acc)
        # writer.add_scalar("Test/acc",acc,epoch)
        vote_pred = vote_pred + predlist 
        vote_score = vote_score + scorelist 
        if epoch % votenum == 0:
            
            # major vote
            vote_pred[vote_pred <= (votenum/2)] = 0
            vote_pred[vote_pred > (votenum/2)] = 1
            vote_score = vote_score/votenum
            
            print('vote_pred', vote_pred)
            print('targetlist', targetlist)
            TP = ((vote_pred == 1) & (targetlist == 1)).sum()
            TN = ((vote_pred == 0) & (targetlist == 0)).sum()
            FN = ((vote_pred == 0) & (targetlist == 1)).sum()
            FP = ((vote_pred == 1) & (targetlist == 0)).sum()
            
            
            print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
            print('TP+FP',TP+FP)
            p = TP / (TP + FP)
            print('precision',p)
            r = TP / (TP + FN)
            print('recall',r)
            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            print('F1',F1)
            print('acc',acc)
            AUC = roc_auc_score(targetlist, vote_score)
            print('AUCp', roc_auc_score(targetlist, vote_pred))
            print('AUC', AUC)
            
            
            if epoch == total_epoch:
                print("saving model...",modelname)
                torch.save(model, "model_backup/"+modelname+"_fire.pth")  # 保存模型


            print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}\n'.format(
            epoch, r, p, F1, acc, AUC))

            f = open('model_result/{}.txt'.format(modelname), 'a+')
            f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}\n'.format(
            epoch, r, p, F1, acc, AUC))
            f.close()
    
    
    # model = DenseNetModel().cuda()
    # modelname = 'DenseNet_medical'


    # ### ResNet18
    # import torchvision.models as models
    # model = models.resnet18(pretrained=True).cuda()
    # modelname = 'ResNet18'


    # ### Dense121
    # import torchvision.models as models
    # model = models.densenet121(pretrained=True).cuda()
    # # modelname = 'Dense121'
    # model.load_state_dict(torch.load('model_backup/Dense121.pt'))
    # modelname = 'Dense121_test'

    # ### Dense169
    # import torchvision.models as models
    # model = models.densenet169(pretrained=True).cuda()
    # modelname = 'Dense169'

    ### Resnet50
    # import torchvision.models as models
    # model = models.resnet50(pretrained=True).cuda()
    # # modelname = 'ResNet50'
    # model.load_state_dict(torch.load('model_backup/ResNet50.pt'))
    # modelname = 'ResNet50_test'
    
    # test_demo()
    # ### VGGNet
    # import torchvision.models as models
    # model = models.vgg16(pretrained=True)
    # model = model.cuda()
    # modelname = 'vgg16'


    # In[139]:


    # ### efficientNet

    # draw_train(total_epoch)