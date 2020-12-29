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


def test_analysis(file_path, true):
    class_name = "fire" if true else "nonfire"
    target_path = "hard_example/"+class_name
    
    print(class_name)
    files = os.listdir(file_path)
    count =0 
    for img in files:
        print(img)
        source_path = file_path+'/'+img
        img = cv2.imread(source_path)  # 读取要预测的图片
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
        img = trans(img)
        img = img.to(device)
        img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
        output = model(img)
        prob = F.softmax(output, dim=1)  # prob是2个分类的概率
        # print(prob)
        value, predicted = torch.max(output.data, 1)
        print(predicted.item())
        # print(value)
        pred_class = classes[predicted.item()]
        print(pred_class)
        if predicted.item() != true:
            #复制文件
            count+=1
            # shutil.copy(source_path,target_path)
    print(count)
    print(len(files))

classes = ('None', 'Fire')
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('Resnet18_fire_with_hard.pth')  # 加载模型
    model = model.to(device)
    
    # 显示网络结构
    # x=torch.randn(10,3,224,224).requires_grad_(True).to(device)
    # y=model(x)
    # vis_graph=make_dot(y)
    # vis_graph.view()
    # 显示模型参数
    input_rd = torch.randn(1,3,224,224)
    flops,params = profile(model,inputs=(input_rd))
    # model.eval()  # 把模型转为test模式
 
    # file_path = "dataset/sun/val"
    # test_analysis(file_path,1)
    ################### 原test ##############
    # for img in files:
    #     print(img)
    #     img = cv2.imread(file_path+'/'+img)  # 读取要预测的图片
    #     img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     trans = transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #     ])
    
    #     img = trans(img)
    #     img = img.to(device)
    #     img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    #     output = model(img)
    #     prob = F.softmax(output, dim=1)  # prob是2个分类的概率
    #     # print(prob)
    #     value, predicted = torch.max(output.data, 1)
    #     print(predicted.item())
    #     # print(value)
    #     pred_class = classes[predicted.item()]
    #     print(pred_class)