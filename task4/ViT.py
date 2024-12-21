import torch
import torch.nn as nn
import torch.optim as optim
import math

from lenet import evaluate, train
from utils import TinySegData, evaluate_result, labels_processor

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device Detected: ", device)

ISDEBUG = False
def debug(*args, **kwargs):
  if ISDEBUG:
    print(*args, **kwargs)

class VisionEmbedding(nn.Module):
  def __init__(self, img_size, in_channels, embed_size, patch_size):
    super(VisionEmbedding, self).__init__()
    self.img_size = img_size
    self.in_channels = in_channels
    self.embed_size = embed_size
    self.patch_size = patch_size
    self.num_patchs = (img_size // patch_size)**2

    self.split = nn.Conv2d(in_channels, embed_size, kernel_size = patch_size, stride = patch_size)
    self.pos_embed = torch.zeros(self.num_patchs + 1, embed_size)

    pos = torch.arange(0, self.num_patchs + 1, step = 1, dtype = dtype)
    pos = pos.unsqueeze(1)
    ia = torch.exp(torch.arange(0, embed_size, step = 2, dtype = dtype) * (-math.log(10000.0) / embed_size))
    ia = ia.unsqueeze(0)

    self.pos_embed[:, 0::2] = torch.sin(pos @ ia)[:, :embed_size // 2 + 1]
    self.pos_embed[:, 1::2] = torch.cos(pos @ ia)[:, :embed_size // 2]

  def forward(self, x):
    x = self.split(x)
    x = x.flatten(2)
    x = x.transpose(1, 2)
    x = torch.cat((torch.zeros(x.shape[0], 1, self.embed_size), x), dim = 1)
    x = x + self.pos_embed
    return x
  
class MultiheadSelfAttention(nn.Module):
  def __init__(self, embed_size, num_heads, dropout_ratio = 0.2):
    super(MultiheadSelfAttention, self).__init__()
    assert(embed_size % num_heads == 0)
    self.embed_size = embed_size
    self.num_heads = num_heads

    self.head_dim = embed_size // num_heads
    self.scale = self.head_dim**-0.5
    self.qkvTrans = nn.Linear(embed_size, embed_size * 3, bias = False)
    self.attnDrop = nn.Dropout(dropout_ratio)

    self.proj = nn.Linear(embed_size, embed_size)

  def forward(self, x):
    B, N, D = x.shape
    qkv = self.qkvTrans(x)
    qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = torch.softmax(attn, dim = -1)
    attn = self.attnDrop(attn)
    x = (attn @ v).transpose(1, 2)
    x = x.reshape(B, N, D)
    x = self.proj(x)
    return x

class MultiLayerPerceptron(nn.Module):
  def __init__(self, embed_size, mlp_size, dropout_ratio = 0.5):
    