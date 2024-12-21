import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import TinySegData, labels_processor, evaluate_result

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device Detected: %s" % device)

ISDEBUG = False
def debug(*args, **kwargs):
  if ISDEBUG:
    print(*args, **kwargs)

class LeNet(nn.Module):
  def __init__(self, in_channels = 3, img_size = 256, num_classes = 5):
    super(LeNet, self).__init__()
    self.model = nn.Sequential(
      # B x 3 x img_size x img_size
      nn.Conv2d(in_channels = in_channels, out_channels = 6, kernel_size = 5, padding = 2),
      # nn.ReLU(),
      nn.MaxPool2d(2, 2),
      # B x 6 x img_size//2 x img_size//2
      nn.Conv2d(6, 16, kernel_size = 4, stride = 4),
      # B x 16 x img_size//2**3 x img_size//2**3
      # nn.ReLU(),
      nn.MaxPool2d(2, 2),
      # B x 16 x img_size//2**4 x img_size//2**4
      nn.Conv2d(16, 32, kernel_size = 5, padding = 2),
      # nn.ReLU(),
      nn.MaxPool2d(2, 2),
      # B x 32 x img_size//2**5 x img_size//2**5
      nn.Conv2d(32, 120, kernel_size = img_size // 2**5),
      nn.Flatten(1),
      # nn.Linear(256, 120),
      nn.ReLU(),
      nn.Linear(120, 84),
      nn.ReLU(),
      nn.Linear(84, num_classes),
      nn.ReLU(),
    )
  def forward(self, x):
    return self.model(x)
  
def train(model, criterion, optimizer, trainset, iteration, batch_size = 50):
  model.train()
  loader = DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 4)
  for i in range(iteration):
    print("Iteration: %d" % (i + 1))
    for d, (features, labels, _) in enumerate(loader, 0):
      optimizer.zero_grad()
      features = features.to(device)
      labels = labels.to(device)

      labels = labels_processor(labels, 5).to(device)
      outputs = model(features)
      outputs = torch.softmax(outputs, dim = 1)

      for i in range(labels.shape[0]):
        if labels[i].sum() < 1e-4:
          labels[i] = torch.zeros(5)
        else:
          labels[i] = labels[i] / labels[i].sum()

      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()

      if (d + 1) % 20 == 0:
        debug("outputs[0]: ", outputs[0])
        debug("answers[0]: ", labels[0])
        print("Data Round %d: Loss = %f" % (d + 1, loss.item()))

def evaluate(model, testset, batch_size = 50):
  model.eval()
  loader = DataLoader(testset, batch_size = batch_size, shuffle = True, num_workers = 4)
  correct = 0
  testset_size = len(testset)
  with torch.no_grad():
    for _, (features, labels, _) in enumerate(loader, 0):
      features = features.to(device)
      labels = labels.to(device)
      labels = labels_processor(labels, 5).to(device)
      outputs = model(features)

      for i in range(labels.shape[0]):
        if labels[i].sum() < 1e-4:
          labels[i] = torch.zeros(5)
        else:
          labels[i] = labels[i] / labels[i].sum()

      correct += evaluate_result(outputs, labels, 1e-4)
  return float(correct) / testset_size

if __name__ == '__main__':
  img_size = (int)(256)

  trainset = TinySegData(phase = 'train', img_size = img_size)
  testset = TinySegData(phase = 'val', img_size = img_size)

  model = LeNet(3, img_size, 5).to(device)
  print("Model Structure:\n%s" % model)

  criterion = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr = 0.01)

  last_ratio = 0.0
  epoch = 0
  iteration = 10
  while True:
    epoch += 1
    print("-----> Epoch: %d" % epoch)
    train(model, criterion, optimizer, trainset, iteration)
    if epoch % 5 == 0:
      ratio = evaluate(model, testset)
      print("<----- Epoch %d: Correct Ratio = %f ----->" % (epoch, ratio))
      if ratio < last_ratio:
        break
      last_ratio = ratio
  print("-----> Final Correct Ratio = %f <-----" % last_ratio)
