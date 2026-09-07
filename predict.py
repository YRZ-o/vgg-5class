from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from net import vgg16
test_pth = r'.\train\testimage\test25.jpg'# 设置可以检测的图像
test = Image.open(test_pth)
'''处理图片'''
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# 对图像预处理，创建了一个变换序列，其中Compose是一个函数，用于将多个图像变换操作组合成一个单一的可调用对象
# 它将 PIL 图像或 NumPy 数组转换为 PyTorch 的 Tensor，并将像素值从 [0, 255] 范围标准化到 [0, 1] 范围。
image = transform(test)# 变换后的图像 image 将是一个 PyTorch Tensor，可以直接用于模型的输入。
'''加载网络'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# CPU与GPU的选择
net = vgg16()# 输入网络
model = torch.load(r".\5catdogchikenhorsebutterfly.pth", map_location=device)# 已训练完成的结果权重输入
net.load_state_dict(model)# 模型导入
net.eval()# 设置为推测模式
image = torch.reshape(image, (1, 3, 224, 224))# 四维图形，RGB三个通
with torch.no_grad():# 不需要计算梯度
    out = net(image)
out = F.softmax(out,dim=1)# softmax 函数确定范围
out = out.data.cpu().numpy()# .data 是用于获取 Tensor 的原始数据（不包含任何操作信息）。
# .cpu() 是将 Tensor 从可能的 GPU 内存移动到 CPU 内存。
# .numpy() 是将 PyTorch Tensor 转换为 NumPy 数组，这样就可以使用 NumPy 的功能。
print(out)
a = int(out.argmax(1))# 输出最大值位置
plt.figure()# 创建一个新的图形窗口。
list = ['Cat','Dog','Chiken','horse','butterfly']
plt.suptitle("Classes:{}:{:.1%}".format(list[a], out[0, a]))# 输出最大概率的道路类型
plt.imshow(test)# 显示输入图像 test
plt.show()# 显示图形窗口
