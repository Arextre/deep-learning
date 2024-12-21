import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

from lenet import train, evaluate
from utils import TinySegData

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device Detected: ", device)

if __name__ == '__main__':
  img_size = 128
  trainset = TinySegData(db_root = "TinySeg", img_size = img_size, phase = 'train')
  testset = TinySegData(db_root = "TinySeg", img_size = img_size, phase = 'val')

  model = resnet18(num_classes = 5).to(device)
  print("model structure:\n%s", model)

  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr = 0.01)
  iteration = 10

  last_ratio = 0.0
  epoch = 0
  while True:
    epoch += 1
    print("-----> Epoch %d start!" % epoch)
    train(model, criterion, optimizer, trainset, iteration)
    if epoch % 5 == 0:
      ratio = evaluate(model, testset)
      print("Epoch: %d, Correct ratio: %f" % (epoch, ratio))
      if last_ratio > ratio:
        break
      last_ratio = ratio

  print("<----- Final Correct Ratio: %f ----->" % last_ratio)