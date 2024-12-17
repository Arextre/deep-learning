# Model: LeNet-5: input -> Conv -> Pool -> Conv -> Pool -> FC -> FC

import torch
import torchvision.datasets as dsets
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ISDEBUG = True
def debug(*args, **kwargs):
  if ISDEBUG:
    print(*args, **kwargs)

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.model = nn.Sequential(
      nn.Conv2d(1, 6, 5),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(6, 16, 5),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(16, 120, 4),
      nn.Flatten(),
      nn.Linear(120, 84),
      nn.Linear(84, 10)
    )
  def forward(self, x):
    return self.model(x)


def train(model, criterion, optimizer, trainset, iteration):
  model.train()
  # debug(">>>>> Trainset size: ", len(trainset))
  batch_size = 100
  loader = DataLoader(trainset, batch_size = batch_size, shuffle = True)

  for i in range(iteration):
    for d, (features, labels) in enumerate(loader, 0):
      optimizer.zero_grad()
      features = features.to(device)
      labels = labels.to(device)

      output = model.forward(features)
      loss = criterion(output, labels)

      loss.backward()
      optimizer.step()

      if (d + 1) % 100 == 0:
        print("Iteration: %d, Loss: %f" % (d + 1, loss.item()))

def evaluate(model, testset):
  model.eval()
  error = 0
  testset_size = len(testset)
  loader = DataLoader(testset, batch_size = 1, shuffle = True)
  with torch.no_grad():
    model.eval()
    for i, (feature, label) in enumerate(loader, 0):
      feature = feature.to(device)
      label = label.to(device)

      output = model.forward(feature)
      _, predict = torch.max(output, 1)
      if predict != label:
        error += 1

  return float(error) / testset_size



if __name__ == '__main__':

  transform = transforms.Compose([
    transforms.ToTensor()
  ])

  trainset = dsets.MNIST(root = './data', train = True, download = True, transform = transform)
  testset = dsets.MNIST(root = './data', train = False, download = True, transform = transform)

  model = LeNet().to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr = 0.001)

  last_ratio = 1.0
  iteration = 2
  epoch = 0

  while True:
    epoch += 1
    print("-----> When epoch = %d" % epoch)
    train(model, criterion, optimizer, trainset, iteration)
    if epoch % 5 == 0:
      ratio = evaluate(model, testset)
      print("Epoch: %d, Error ratio: %f" % (epoch, ratio))
      if last_ratio < ratio:
        break
      else:
        last_ratio = ratio
  
  print("<----- Final error ratio: %f ----->" % (last_ratio))