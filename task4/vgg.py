import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import TinySegData
from lenet import train, evaluate

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device Detected: %s" % device)

class VGG(nn.Module):
  def __init__(self, config, cls_size = 2048, img_channels = 3, img_size = 256, num_classes = 5):
    super(VGG, self).__init__()

    layers = []
    in_channels = img_channels
    Mcnt = 0
    for v in config:
      if v == 'M':
        layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
        Mcnt += 1
      else:
        layers.append(nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1))
        layers.append(nn.ReLU())
        in_channels = v
    
    assert(img_size % 2**Mcnt == 0)
      
    self.features = nn.Sequential(*layers)
    img_size = img_size // 2**Mcnt
    self.cls = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_channels * img_size**2, cls_size),
      nn.ReLU(),
      nn.Dropout(p = 0.5),
      nn.Linear(cls_size, cls_size),
      nn.ReLU(),
      nn.Linear(cls_size, num_classes)
    )
  
  def forward(self, x):
    x = self.features(x)
    x = self.cls(x)
    return x

if __name__ == '__main__':
  img_size = 256
  configs = {
    "TinyVGG": [32, 'M', 64, 'M'],
    "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
  }
  
  trainset = TinySegData(img_size = img_size, phase = 'train')
  testset = TinySegData(img_size = img_size, phase = 'val')
  
  modelVer = 'TinyVGG'

  model = VGG(configs[modelVer], cls_size = 256, img_channels = 3, img_size = img_size, num_classes = 5).to(device)
  print("Model Structure:\n%s" % model)

  criterion = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr = 0.01)

  last_ratio = 0.0
  epoch = 0
  iteration = 10
  while True:
    epoch += 1
    print("-----> Epoch: %d" % epoch)
    train(model, criterion, optimizer, trainset, iteration, batch_size = 50)
    if epoch % 5 == 0:
      ratio = evaluate(model, testset, batch_size = 50)
      print("<----- Epoch: %d, Correct Ratio: %f ----->" % (epoch, ratio))
      if ratio < last_ratio:
        break
      last_ratio = ratio

  print("-----> Final Correct Ratio: %f" % last_ratio)