from PIL import Image
import os

# 指定JPEG文件所在的目录
input_dir = 'E:\\AAA_CV\\vgg_dogclass\\train\\horse1'
output_dir = 'E:\\AAA_CV\\vgg_dogclass\\train\\horse'
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历JPEG文件并转换为JPG
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.jpeg'):
        img = Image.open(os.path.join(input_dir, filename))
        new_filename = os.path.join(output_dir, filename[:-5] + '.jpg')
        img.save(new_filename)
