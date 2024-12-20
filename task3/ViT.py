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

print("Device Detected: ", device)

ISDEBUG = False
def debug(*args, **kwargs):
  if ISDEBUG:
    print(*args, **kwargs)

class VisionEmbedding(nn.Module):
  def __init__(self, img_size, in_channels, embedding_size, patch_size):
    super(VisionEmbedding, self).__init__()
    assert(img_size % patch_size == 0)
    self.img_size = img_size
    self.in_channels = in_channels
    self.embedding_size = embedding_size
    self.patch_size = patch_size
    self.num_patchs = (img_size // patch_size)**2 # number of patchs

    # split the image into patchs
    self.split = nn.Conv2d(in_channels, embedding_size, kernel_size = patch_size, stride = patch_size)
    self.pos_embed = torch.zeros(self.num_patchs + 1, embedding_size)

    # positional encoding
    pos = torch.arange(0, self.num_patchs + 1, step = 1, dtype = dtype)
    pos = pos.unsqueeze(1) # from (num_patch+1) Dim vector to (num_patch+1)x1 matrix
    ia = torch.exp(torch.arange(0, embedding_size, step = 2, dtype = dtype) * (-math.log(10000.0) / embedding_size))
    ia = ia.unsqueeze(0)  # to matrix
    debug("pos shape: ", pos.shape)
    debug("ia shape: ", ia.shape)
    self.pos_embed[:, 0::2] = torch.sin(pos @ ia)[:, :embedding_size // 2 + 1]
    self.pos_embed[:, 1::2] = torch.cos(pos @ ia)[:, :embedding_size // 2]

  def forward(self, x: torch.Tensor):
    # (batch, in_channels, size, size)
    x = self.split(x)
    # (B, embedding_size, ...)
    x = x.flatten(2)
    # (B, embedding_size, num_patch)
    x = x.transpose(1, 2)
    # (B, num_patch, D)
    x = torch.cat((torch.zeros(x.shape[0], 1, self.embedding_size), x), dim = 1)
    # (B, num_patch+1, D), inserted a class token
    x = x + self.pos_embed
    return x


class MultiheadSelfAttention(nn.Module):
  def __init__(self, embedding_size, num_heads, dropout_ratio = 0.5):
    super(MultiheadSelfAttention, self).__init__()
    assert(embedding_size % num_heads == 0) # should be divided
    self.embedding_size = embedding_size
    self.num_heads = num_heads

    self.head_dim = embedding_size // num_heads
    self.scale = self.head_dim ** -0.5 # scale factor
    self.qkvTrans = nn.Linear(embedding_size, embedding_size * 3, bias = False) # no bias added
    self.attnDrop = nn.Dropout(dropout_ratio)

    self.proj = nn.Linear(embedding_size, embedding_size)

  def forward(self, x):
    B, N, D = x.shape
    # (B, N, D), N = num_patchs + 1
    qkv = self.qkvTrans(x)
    # (B, N, 3D)
    qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
    # (B, N, 3, heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    # (B, heads, N, head_dim)
    attn = (q @ k.transpose(-2, -1)) * self.scale
    # (B, heads, N, N)
    attn = attn.softmax(dim = -1)
    attn = self.attnDrop(attn)
    x = (attn @ v).transpose(1, 2)
    # (B, N, heads, head_dim)
    x = x.reshape(B, N, D)
    x = self.proj(x)
    # (B, N, D)
    return x

MSA = MultiheadSelfAttention

class TransformerEncoder(nn.Module):
  def __init__(self, embedding_size, num_heads, mlp_size, dropout_ratio = 0.5):
    super(TransformerEncoder, self).__init__()

    self.embedding_size = embedding_size
    self.num_heads = num_heads
    self.mlp_size = mlp_size
    self.dropout_ratio = dropout_ratio

    self.normMSA = nn.LayerNorm(embedding_size)
    self.MSA = MSA(embedding_size, num_heads, dropout_ratio)

    self.normMLP = nn.LayerNorm(embedding_size)
    self.MLP = nn.Sequential(
      nn.Linear(embedding_size, mlp_size),
      nn.GELU(),
      nn.Dropout(dropout_ratio),
      nn.Linear(mlp_size, embedding_size),
      nn.Dropout(dropout_ratio)
    )

  def forward(self, x):
    x = x + self.MSA(self.normMSA(x))
    x = x + self.MLP(self.normMLP(x))
    return x




class ViT(nn.Module):
  # input should be a 4D tensor (batch, channels, size, size)
  def __init__(self, img_size, in_channels, patch_size, embedding_size, num_heads, mlp_size, num_classes, depth, strict_mode = True):
    super(ViT, self).__init__()
    if strict_mode:
      assert(in_channels * patch_size**2 == embedding_size,
        "ViT Strict Mode Error: in_channels * patch_size^2 should be equal to embedding_size"
      )
    assert(img_size % patch_size == 0) # should be divided

    self.img_size = img_size
    self.in_channels = in_channels
    self.embedding_size = embedding_size
    self.patch_size = patch_size
    self.num_heads = num_heads
    self.mlp_size = mlp_size
    self.num_classes = num_classes
    self.depth = depth
    
    self.embedding_layer = VisionEmbedding(img_size, in_channels, embedding_size, patch_size)
    self.encoders = nn.ModuleList([
      TransformerEncoder(embedding_size, num_heads, mlp_size, dropout_ratio = 0.5) for _ in range(depth)
    ])

    self.norm = nn.LayerNorm(embedding_size)
    self.classifier = nn.Linear(embedding_size, num_classes)
  
  def forward(self, x):
    # (batch, channels, size, size)
    x = self.embedding_layer(x)
    # (batch, num_patch+1, embedding_size)
    for encoder in self.encoders:
      x = encoder(x)
    # (B, N, D)
    x = self.norm(x)
    # (B, N, D)
    x = self.classifier(x[:, 0])
    # (B, num_classes)
    return x

  

if __name__ == '__main__':
  modelParam = {
    "TinyViT": (28, 1, 7, 49, 7, 128, 10, 3),
    "StdViT": (224, 3, 16, 768, 12, 2048, 10, 12),
    "TestViT": (112, 1, 14, 196, 14, 256, 10, 3),
  }
  imgTransformer = {
    "TinyViT": transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean = (0.5,), std = (0.5,))
    ]),
    "StdViT": transforms.Compose([
      transforms.Resize(224),
      transforms.Grayscale(num_output_channels = 3),
      transforms.ToTensor(),
      transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ]),
    "TestViT": transforms.Compose([
      transforms.Resize(112),
      transforms.ToTensor(),
      transforms.Normalize(mean = (0.5,), std = (0.5,))
    ]),
  }

  modelVer = 'TestViT'

  model = ViT(*modelParam[modelVer]).to(device)
  trainset = dsets.MNIST(root = './data', train = True, download = True, transform = imgTransformer[modelVer])
  testset = dsets.MNIST(root = './data', train = False, download = True, transform = imgTransformer[modelVer])
  print(model)
  
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
      print("<----- Epoch %d, Error ratio: %f ----->" % (epoch, ratio))
      if last_ratio < ratio:
        break
      else:
        last_ratio = ratio
  
  print("-----> Final Ratio = %f" % last_ratio)