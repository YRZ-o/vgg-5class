import torch
import torchvision

print(torch.__version__)
print(torchvision.__version__)

x = 255
x /= 175.5 # 0.45299145299145294
# x -= 1.
print(x)
