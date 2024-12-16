import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
from torchvision import transforms

class VGG(nn.Module):
  def __init__(self, config, num_classes = 10):
    super(VGG, self).__init__()

    layers = []
    in_channels = 3
    for v in config:
      if v == 'M':
        layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
      else:
        layers.append(nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        in_channels = v # update the in_channels
    
    self.features = nn.Sequential(*layers)
    self.classifies = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512 * 7 * 7, 2048),
      nn.ReLU(inplace = True),
      nn.Dropout(p = 0.5),
      nn.Linear(2048, 2048),
      nn.ReLU(inplace = True),
      nn.Linear(2048, num_classes)
    )
  
  def forward(self, x):
    x = self.features(x)
    x = self.classifies(x)
    return x

if __name__ == '__main__':
  imageTransformer = transforms.Compose([
    transforms.Resize(224), # resize the image to 224x224
    transforms.Grayscale(num_output_channels = 3), # vgg requires 3 channels input
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)) # normalize the input
  ])
  trainset = dsets.MNIST(root = './data', train = True, download = True, transform = imageTransformer)
  testset = dsets.MNIST(root = './data', train = False, download = True, transform = imageTransformer)

  # Hyperparameters: VGG-16
  config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

  model = VGG(config)