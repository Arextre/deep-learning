import torch
import torchvision
from torch.utils.data import Dataset
import cv2
from PIL import Image
import numpy as np
import random
random.seed(0)

readvdnames = lambda x: open(x).read().rstrip().split('\n')

class TinySegData(Dataset):
  def __init__(self, db_root="TinySeg", img_size=256, phase='train'):
    classes = ['person', 'bird', 'car', 'cat', 'plane', ]
    seg_ids = [1, 2, 3, 4, 5]

    templ_image = db_root + "/JPEGImages/{}.jpg"
    templ_mask = db_root + "/Annotations/{}.png"

    ids = readvdnames(db_root + "/ImageSets/" + phase + ".txt")

    # build training and testing dbs
    samples = []
    for i in ids:
      samples.append([templ_image.format(i), templ_mask.format(i)])
    self.samples = samples
    self.phase = phase
    self.db_root = db_root
    self.img_size = img_size

    self.color_transform = torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2)

    if not self.phase == 'train':
      print ("resize and augmentation will not be applied...")

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    if self.phase == 'train':
      return self.get_train_item(idx)
    else:
      return self.get_test_item(idx)

  def get_train_item(self, idx):
    sample = self.samples[idx]
    image = Image.open(sample[0])

    if random.randint(0, 1) > 0:
      image = self.color_transform(image)
    image = np.asarray(image)[..., ::-1]     # to BGR
    seg_gt = (np.asarray(Image.open(sample[1]).convert('P'))).astype(np.uint8)

    image = image.astype(np.float32)
    image = image / 127.5 - 1        # -1~1

    if random.randint(0, 1) > 0:
      image = image[:, ::-1, :]       # HWC
      seg_gt = seg_gt[:, ::-1]

    # random crop to 256x256
    height, width = image.shape[0], image.shape[1]
    if height == width:
      miny, maxy = 0, 256
      minx, maxx = 0, 256
    elif height > width:
      miny = np.random.randint(0, height-256)
      maxy = miny+256
      minx = 0
      maxx = 256
    else:
      miny = 0
      maxy = 256
      minx = np.random.randint(0, width-256)
      maxx = minx+256
    image = image[miny:maxy, minx:maxx, :].copy()
    seg_gt = seg_gt[miny:maxy, minx:maxx].copy()

    if self.img_size != 256:
      new_size = (self.img_size, self.img_size)
      image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
      seg_gt = cv2.resize(seg_gt, new_size, interpolation=cv2.INTER_NEAREST)

    image = np.transpose(image, (2, 0, 1))      # To CHW

    # cv2.imwrite("test.png", np.concatenate([(image[0]+1)*127.5, seg_gt*255], axis=0))
    return image, seg_gt, sample

  def get_test_item(self, idx):
    sample = self.samples[idx]
    image = cv2.imread(sample[0])
    seg_gt = (np.asarray(Image.open(sample[1]).convert('P'))).astype(np.uint8)
    image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR).copy()
    seg_gt = cv2.resize(seg_gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST).copy()
    image = image.astype(np.float32)
    image = image / 127.5 - 1  # -1~1
    image = np.transpose(image, (2, 0, 1))
    return image, seg_gt, sample

def labels_processor(labels, num_classes):
  assert(len(labels.shape) == 3) # (batch_size, img_size, img_size)
  labels_after = torch.zeros(labels.shape[0], num_classes)
  for b in range(labels.shape[0]):
    val = labels[b, :, :].view(-1)
    uniq_val, uniq_cnt = torch.unique(val, return_counts = True)
    for val, cnt in zip(uniq_val, uniq_cnt):
      if val != 0:
        labels_after[b, val.item() - 1] = cnt.item()
  return labels_after

def evaluate_result(outputs, labels, eps = 1e-1):
  assert(len(outputs.shape) == 2) # (batch_size, num_classes + 1)
  assert(outputs.shape == labels.shape)
  ret = outputs.shape[0]
  for b in range(outputs.shape[0]):
    for c in range(outputs.shape[1]):
      if labels[b, c].item() > eps and outputs[b, c].item() < eps:
        ret -= 1 # class loss
        break
      if labels[b, c].item() < eps and outputs[b, c].item() > eps:
        ret -= 1 # class gain
        break
  return ret