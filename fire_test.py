import torch
import cv2
import os
import torch.nn.functional as F
from CatDog import Net  ##重要，虽然显示灰色(即在次代码中没用到)，但若没有引入这个模型代码，加载模型时会找不到模型
from torchvision import datasets, transforms
from PIL import Image
from torchviz import make_dot
import shutil
from thop import profile

trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

def test_analysis(file_path, true):
    class_name = "positive" if true else "negative"
    target_path = "hard_example/"+class_name
    
    print(class_name)
    files = os.listdir(file_path)
    count =0 
    for img in files:
        print(img)
        source_path = file_path+'/'+img
        img = cv2.imread(source_path)  # 读取要预测的图片
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
    
        img = trans(img)
        img = img.to(device)
        img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
        output = model(img)
        prob = F.softmax(output, dim=1)  # prob是2个分类的概率
        # print(prob)
        value, predicted = torch.max(output.data, 1)
        print(predicted.item())
        # print(value)
        if predicted.item():
            # pred_class = classes[predicted.item()]
            pred_class = "Fire"
        else:
            pred_class = "Non"
        
        print(pred_class)
        if predicted.item() != true:
            #复制文件
            count+=1
            shutil.copy(source_path,target_path)
    print(count)
    print(len(files))

def test_demo(file_path):
    TP,TN,FP,FN=0,0,0,0
    pos_files = os.listdir(file_path+'/positive')
    neg_files = os.listdir(file_path+'/negative')
    # 把模型转为test型    
    model.eval()
    # 对fire样本，标签为1
    for img in pos_files:
        print(img)
        source_path = file_path+'/positive/'+img
        img = cv2.imread(source_path)  # 读取要预测的图片
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
        img = trans(img)
        img = img.to(device)
        img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
        output = model(img)
        prob = F.softmax(output, dim=1)  # prob是2个分类的概率

        value, predicted = torch.max(output.data, 1)
        print(predicted.item())
        # print(value)
        pred_class = classes[predicted.item()]
        print(pred_class)
        # 对fire样本
        if predicted.item() != 1:
            FP += 1
        else:
            TP += 1
    # 对nonfire样本,标签为0
    for img in neg_files:
        print(img)
        source_path = file_path+'/negative/'+img
        img = cv2.imread(source_path)  # 读取要预测的图片
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
        img = trans(img)
        img = img.to(device)
        img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
        output = model(img)
        prob = F.softmax(output, dim=1)  # prob是2个分类的概率

        value, predicted = torch.max(output.data, 1)
        # print(predicted.item())
        pred_class = classes[predicted.item()]
        print(pred_class)
        if predicted.item() != 0:
            FN += 1
        else:
            TN += 1

    print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
    print('TP+FP:',TP+FP)
    p = TP / (TP + FP)
    print('precision:',p)
    r = TP / (TP + FN)
    print('recall:',r)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('F1:',F1)
    print('acc:',acc)

classes = ('None', 'Fire')
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model_backup/ResNet50_fire.pth')  # 加载模型
    model = model.to(device)
    
    # 显示网络结构
    # x=torch.randn(10,3,224,224).requires_grad_(True).to(device)
    # y=model(x)
    # vis_graph=make_dot(y)
    # vis_graph.view()
    # 显示模型参数
    # input_rd = torch.randn(1,3,224,224)
    # flops,params = profile(model,inputs=(input_rd))
    model.eval()  # 把模型转为test模式
 
    file_path = "dataset/sun/train"
    test_demo(file_path)
    