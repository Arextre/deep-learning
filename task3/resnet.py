import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from lenet import train, evaluate

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device Detected: ", device)

ISDEBUG = False
def debug(*args, **kwargs):
  if ISDEBUG:
    print(*args, **kwargs)

if __name__ == '__main__':
  transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
  ])
  trainset = dsets.MNIST(root = './data', train = True, download = True, transform = transforms)
  testset = dsets.MNIST(root = './data', train = False, download = True, transform = transforms)

  model = resnet18(num_classes = 10).to(device)
  # rewrite the first layer to accept 1 channel
  model.conv1 = nn.Conv2d(1, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False).to(device)
  debug("Model structure:\n", model)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr = 0.0005)

  last_ratio = 1.0
  iteration = 1
  epoch = 0

  while True:
    epoch += 1
    print("-----> Epoch %d start!" % epoch)
    train(model, criterion, optimizer, trainset, iteration)
    if epoch % 5 == 0:
      ratio = evaluate(model, testset)
      print("Epoch: %d, Error ratio: %f" % (epoch, ratio))
      if last_ratio < ratio:
        break
      else:
        last_ratio = ratio

  print("<----- Final ratio: %f ----->" % last_ratio)