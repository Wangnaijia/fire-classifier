import os
import imgaug as ia
import random
import numpy as np
import imgaug.augmenters as iaa
import cv2

def rename(path):
    '该文件夹下所有的文件（包括文件夹）'
    FileList = os.listdir(path)
    '遍历所有文件'
    for files in FileList:
        '原来的文件路径'
        oldDirPath = os.path.join(path, files)
        '如果是文件夹则递归调用'
        if os.path.isdir(oldDirPath):
            rename(oldDirPath)
        '文件名'
        fileName = os.path.splitext(files)[0]
        '文件扩展名'
        fileType = os.path.splitext(files)[1]
        '新的文件路径'
        newDirPath = os.path.join(path, "sun"+fileName+ fileType)
        '重命名'
        os.rename(oldDirPath, newDirPath)

"""
	image_augmentation.py
"""

'''
定义裁剪函数，四个参数分别是：
左上角横坐标x0
左上角纵坐标y0
裁剪宽度w
裁剪高度h
'''
crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]

'''
随机裁剪
area_ratio为裁剪画面占原画面的比例
hw_vari是扰动占原高宽比的比例范围
'''
def random_crop(img, area_ratio, hw_vari):
    h, w = img.shape[:2]
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta
	
	# 下标进行裁剪，宽高必须是正整数
    w_crop = int(round(w*np.sqrt(area_ratio*hw_mult)))
	
	# 裁剪宽度不可超过原图可裁剪宽度
    if w_crop > w:
        w_crop = w
		
    h_crop = int(round(h*np.sqrt(area_ratio/hw_mult)))
    if h_crop > h:
        h_crop = h
	
	# 随机生成左上角的位置
    x0 = np.random.randint(0, w-w_crop+1)
    y0 = np.random.randint(0, h-h_crop+1)
	
    return crop_image(img, x0, y0, w_crop, h_crop)

'''
定义旋转函数：
angle是逆时针旋转的角度
crop是个布尔值，表明是否要裁剪去除黑边
'''
def rotate_image(img, angle, crop):
    h, w = img.shape[:2]
	
	# 旋转角度的周期是360°
    angle %= 360
	
	# 用OpenCV内置函数计算仿射矩阵
    M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
	
	# 得到旋转后的图像
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))

	# 如果需要裁剪去除黑边
    if crop:
        angle_crop = angle % 180             	    # 对于裁剪角度的等效周期是180°
        if angle_crop > 90:                        	# 并且关于90°对称
            angle_crop = 180 - angle_crop
		
        theta = angle_crop * np.pi / 180.0    		# 转化角度为弧度
        hw_ratio = float(h) / float(w)    		    # 计算高宽比
		

        tan_theta = np.tan(theta)                   # 计算裁剪边长系数的分子项
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
		

        r = hw_ratio if h > w else 1 / hw_ratio		# 计算分母项中和宽高比相关的项
        denominator = r * tan_theta + 1		 		# 计算分母项

        crop_mult = numerator / denominator			# 计算最终的边长系数
        w_crop = int(round(crop_mult*w))			# 得到裁剪区域
        h_crop = int(round(crop_mult*h))
        x0 = int((w-w_crop)/2)
        y0 = int((h-h_crop)/2)
        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)
    return img_rotated

'''
随机旋转
angle_vari是旋转角度的范围[-angle_vari, angle_vari)
p_crop是要进行去黑边裁剪的比例
'''
def random_rotate(img, angle_vari, p_crop):
    angle = np.random.uniform(-angle_vari, angle_vari)
    crop = False if np.random.random() > p_crop else True
    return rotate_image(img, angle, crop)

'''
定义hsv变换函数：
hue_delta是色调变化比例
sat_delta是饱和度变化比例
val_delta是明度变化比例
'''
def hsv_transform(img, hue_delta, sat_mult, val_mult):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)

'''
随机hsv变换
hue_vari是色调变化比例的范围
sat_vari是饱和度变化比例的范围
val_vari是明度变化比例的范围
'''
def random_hsv_transform(img, hue_vari, sat_vari, val_vari):
    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)
    return hsv_transform(img, hue_delta, sat_mult, val_mult)

'''
定义gamma变换函数：
gamma就是Gamma
'''
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

'''
随机gamma变换
gamma_vari是Gamma变化的范围[1/gamma_vari, gamma_vari)
'''
def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)

"""主函数"""
def simple_augmentation(filelist):
    files = os.listdir(filelist)
    for filepath in files:
        print(filepath)
        filename = filepath.split(os.sep)[-1]
        dot_pos = filename.rfind('.')
		
		# 获取文件名和后缀名
        imgname = filename[:dot_pos]
        ext = filename[dot_pos:]

        print('Augmenting {} ...'.format(filename))
        img = cv2.imread(filelist+'/'+filename)
        # for i in range(2):
        img_varied = img.copy()
        
        # 扰动后文件名的前缀
        varied_imgname = '{}_'.format(imgname)
        
        # 按照比例随机对图像进行镜像
        if random.random() < 0.5:
            # 利用numpy.fliplr(img_varied)也能实现
            img_varied = cv2.flip(img_varied, 1)
            varied_imgname += 'm'
        
        # 按照比例随机对图像进行裁剪
        if random.random() < 0.5:
            img_varied = random_crop(
                img_varied,
                0.8,
                0.1)
            varied_imgname += 'c'
        
        # 按照比例随机对图像进行旋转
        if random.random() < 0.5:
            img_varied = random_rotate(
                img_varied,
                1.0,
                10.0)
            varied_imgname += 'r'
        
        # 按照比例随机对图像进行HSV扰动
        if random.random() < 0.5:
            img_varied = random_hsv_transform(
                img_varied,
                10,
                0.1,
                0.1)
            varied_imgname += 'h'
        
        # 按照比例随机对图像进行Gamma扰动
        if random.random() < 0.5:
            img_varied = random_gamma_transform(
                img_varied,
                2.0)
            varied_imgname += 'g'
        
        # 生成扰动后的文件名并保存在指定的路径
        output_filepath = os.sep.join([
            '/home/wnj/Projects/classifier_fire/train_expanded/negative/',
            '{}{}'.format(varied_imgname, ext)])
        cv2.imwrite(output_filepath, img_varied)

path = '/home/wnj/Projects/classifier_fire/dataset/sun/val'
aug_path = '/home/wnj/Projects/classifier_fire/train_expanded/negative/'
simple_augmentation(aug_path)
# rename(path)
    # seq = iaa.Sequential([
    #     #从图片边随机裁剪50~100个像素,裁剪后图片的尺寸和之前不一致
    #     #通过设置keep_size为True可以保证裁剪后的图片和之前的一致
    #     # iaa.Crop(px=(50,100),keep_size=False),
    #     # #50%的概率水平翻转
    #     # iaa.Fliplr(0.5),
    #     # #50%的概率垂直翻转
    #     # iaa.Flipud(0.5),
    #     # #高斯模糊,使用高斯核的sigma取值范围在(0,3)之间
    #     # #sigma的随机取值服从均匀分布
    #     iaa.GaussianBlur(sigma=(0,3.0))
    # ])
    # #可以内置的quokka图片,设置加载图片的大小
    # # example_img = ia.quokka(size=(224,224))
    # #这里我们使用自己的图片
    # example_img = cv2.imread("/home/wnj/Projects/classifier_fire/train_300/negative/479.jpg")
    # #对图片的通道进行转换,由BGR转为RGB
    # #imgaug处理的图片数据是RGB通道
    # example_img = example_img[:,:,::-1]
    # #数据增强,针对单张图片
    # aug_example_img = seq.augment_image(image=example_img)
    # print(example_img.shape,aug_example_img.shape)
    # #(700, 700, 3) (544, 523, 3)
    # #显示图片
    # ia.imshow(aug_example_img)
    # cv2.imwrite("after_aug.jpg",aug_example_img)

# simple_example()


# path = '/home/wnj/Projects/classifier_fire/hard_example/negative'
# rename(path)