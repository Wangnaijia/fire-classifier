# fire-classifier
# 明火分类项目
# 目录结构：
>dataset 数据集
>model_backup 保存模型位置
>model_result 模型结果存放
>runs tensorboard 运行位置
rename.py文件中包含对文件批量重命名与批量数据增强的操作

# 实验步骤：
- 1. 在resnet18上利用train数据集进行训练分类，得到模型modelFire.pth 在tensorboard上画出收敛曲线。如果没有分出验证集，不用val
- 2. 在自己准备的数据集上利用训练好的模型来分类测试，与lable不同的图片作为难例
- 3. 将难例加入到原训练集上进行再次分类训练
                    ResNet18
            |      | train | TP | TN | FN | FP 
positive    | fire | 2003  |1652|    | 351|
negative    | lamp | 412   |    |382 |    | 30
            | sun  | 425   |    |413 |    | 12

            
- 4. 问题：正例和反例比例不平衡，正例477张，反例仅196
        解决： 使用imgaug
            |      | val | TP | TN | FN | FP 
positive    | fire | 170 |169 |    | 1  |
negative    | lamp | 88  |    | 87 |    | 1
            | sun  | 90   |   | 87 |    | 3

- 5. 在backbone上面分别跑了一次分类
    ResNet50>>efficientnet-b0>>vgg16>>Dense169>>resnet18>>dense121

- 6. 在自己准备的dataset上面进行分类测试
                    ResNet50
            |      | train | TP | TN | FN | FP 
positive    | fire | 2003  |1816|    | 187|
negative    | lamp | 412   |    |388 |    | 24
            | sun  | 425   |    |417 |    | 8

- 7. 扩充数据集：
    dataset_280                 >>>>          train_expanded           
      | pos | neg | total |                 | pos | neg | total |
train | 101 | 123 | 224   |            train| 288 | 155 | 443   |
val   | 25  | 31  | 56    |             val | 195 | 209 | 404   |

- 8. finetuning
- 9. 进行验证集上的验证

                  val_pos | val_neg   
                   195    |   209      
                | TP | TN | FN | FP 
ResNet50        | 188 | 200| 7  | 9 
EfficientNet-b0 | 186 | 192| 9  | 17
vgg16           | 182 | 174| 13 | 35 

