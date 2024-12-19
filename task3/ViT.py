import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils.data import DataLoader

from lenet import evaluate, train

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisionEmbedding(nn.Mudule):
  def __init__(self, img_size, in_channels, embedding_size, patch_size):
    super(VisionEmbedding, self).__init__()
    assert(img_size % patch_size == 0)
    self.img_size = img_size
    self.in_channels = in_channels
    self.embedding_size = embedding_size
    self.patch_size = patch_size
    self.N = (img_size // patch_size)**2 # number of blocks
    self.split = nn.Conv2d(in_channels, embedding_size, kernel_size = patch_size, stride = patch_size)
    self.pos_embed = torch.zeros(self.N + 1, embedding_size)
    pos = torch.arange(0, self.N + 1, step = 1, dtype = dtype)
    pos.unsqueeze(1) # from (N+1) Dim vector to (N+1)x1 matrix
    ia = torch.exp(torch.arange(0, embedding_size, step = 2, dtype = dtype) * (-math.log(10000.0) / embedding_size))
    ia.unsqueeze(0)  # to matrix
    self.pos_embed[:, 0::2] = torch.sin(pos * ia)
    self.pos_embed[:, 1::2] = torch.cos(pos * ia)
  def forward(self, x: torch.Tensor):
    x = self.split(x)
    x = x.flatten(2)
    x.transpose(1, 2)
    # (B, N, D)
    x = torch.cat((torch.zeros(1, 1, self.embedding_size), x), dim = 1)
    # (B, N+1, D), inserted a class token
    x = x + self.pos_embed
    return x

class TransformerEncoder(nn.Mudule):
  def __init__(self, embedding_size, num_heads):
    super(TransformerEncoder, self).__init__()
    

class ViT(nn.Module):
  # input should be a 4D tensor (batch, channels, size, size)
  def __init__(self, img_size, in_channels, embedding_size, patch_size, num_heads, num_classes, depth):
    super(ViT, self).__init__()
    assert(in_channels * patch_size**2 == embedding_size)
    assert(img_size % patch_size == 0) # should be divided

    self.img_size = img_size
    self.in_channels = in_channels
    self.embedding_size = embedding_size
    self.patch_size = patch_size
    self.num_heads = num_heads
    self.num_classes = num_classes
    self.depth = depth
    
    self.embedding_layer = VisionEmbedding(img_size, in_channels, embedding_size, patch_size)
    self.encoder = nn.ModuleList([
      TransformerEncoder(embedding_size, num_heads) for _ in range(depth)
    ]) 


if __name__ == '__main__':
  pass