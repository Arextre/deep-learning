import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils.data import DataLoader

from lenet import train, evaluate

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device Detected: ", device)

class VGG(nn.Module):
  def __init__(self, config, classi_size = 2048, imgChannel = 3, num_classes = 10):
    super(VGG, self).__init__()

    layers = []
    in_channels = imgChannel
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
      nn.Linear(in_channels * 7 * 7, classi_size),
      nn.ReLU(inplace = True),
      nn.Dropout(p = 0.5),
      nn.Linear(classi_size, classi_size),
      nn.ReLU(inplace = True),
      nn.Linear(classi_size, num_classes)
    )
  
  def forward(self, x):
    x = self.features(x)
    x = self.classifies(x)
    return x

if __name__ == '__main__':
  """
  imageTransformer = transforms.Compose([
    transforms.Resize(224), # resize the image to 224x224
    transforms.Grayscale(num_output_channels = 3), # vgg requires 3 channels input
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)) # normalize the input
  ])
  """
  imageTransformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5,), std = (0.5,))
  ])
  trainset = dsets.MNIST(root = './data', train = True, download = True, transform = imageTransformer)
  testset = dsets.MNIST(root = './data', train = False, download = True, transform = imageTransformer)

  # Hyperparameters: VGG-16
  # config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

  # a tiny version, for AMD CPU
  config = [32, 'M', 64, 'M']
  model = VGG(config, classi_size = 128, imgChannel = 1, num_classes = 10).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr = 0.0005)

  last_ratio = 1.0
  epoch = 0
  iteration = 1

  while True:
    epoch += 1
    print("-----> Epoch %d start" % epoch)
    train(model, criterion, optimizer, trainset, iteration)
    if epoch % 5 == 0:
      ratio = evaluate(model, testset)
      print("Epoch %d, Error ratio: %f" % (epoch, ratio))
      if last_ratio < ratio:
        break
      else:
        last_ratio = ratio
  
  print("<----- Final Ratio: %f ----->" % last_ratio)