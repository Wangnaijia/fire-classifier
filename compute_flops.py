import torch
from thop import profile
import torchvision.models as models

model = models.resnet18()
# modelname = 'ResNet18'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.load('Resnet18_fire_with_hard.pth')  # 加载模型
# model = model.to(device)

# 显示网络结构
# x=torch.randn(10,3,224,224).requires_grad_(True).to(device)
# y=model(x)
# vis_graph=make_dot(y)
# vis_graph.view()
# 显示模型参数
input_rd = torch.randn(1,3,224,224)
flops,params = profile(model,inputs=(input_rd))