#!/usr/bin/env python3

import numpy as np
from functools import reduce
import sys
import keras
import datetime

DEBUG = True
def debug(*args, **kwargs):
  if DEBUG:
    print(*args, **kwargs, file = sys.stderr)

# Activator types
class SigmoidActivator(object):
  def forward(self, weighted_input):
    return 1.0 / (1.0 + np.exp(-weighted_input))
  def backward(self, output):
    return output * (1 - output)

class TanhActivator(object):
  def forward(self, weighted_input):
    return np.tanh(weighted_input)
  def backward(self, output):
    return 1 - output * output

class SoftmaxActivator(object):
  def forward(self, weighted_input):
    S = np.sum(np.exp(weighted_input))
    return np.exp(weighted_input) / S
  def backward(self, output):
    return output * (1 - output)

# LossFunc types
class CrossEntropyLoss(object):
  def forward(self, output, label):
    return -np.sum(label * np.log(output))
  def backward(self, output, label):
    return label / output

class SquareLoss(object):
  def forward(self, output, label):
    return 0.5 * np.sum((output - label) ** 2)
  def backward(self, output, label):
    return output - label

# Layer types
class FullConnectedLayer(object):
  def __init__(self, input_size, output_size, activator = 'TanhActivator'):
    self.input_size = input_size
    self.output_size = output_size

    if activator == 'TanhActivator':
      self.activator = TanhActivator()
    elif activator == 'SigmoidActivator':
      self.activator = SigmoidActivator()
    elif activator == 'SoftmaxActivator':
      self.activator = SoftmaxActivator()
    else:
      debug('Unkown activator: %s' % (activator))
      sys.exit(1)

    self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
    self.b = np.zeros((output_size, 1))

  def forward(self, input_array):
    assert((len(input_array) == self.input_size))
    self.input = input_array
    self.output = self.activator.forward(np.dot(self.W, input_array) + self.b)

  def backward(self, delta_array):
    assert(len(delta_array) == self.output_size)
    if type(self.activator) == SoftmaxActivator:
      self.predelta = np.dot(self.W.T, delta_array)
    else:
      self.predelta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array)
    self.W_grad = np.dot(delta_array, self.input.T)
    self.b_grad = delta_array

  def update(self, rate):
    self.W += -rate * self.W_grad
    self.b += -rate * self.b_grad

  def dump(self):
    print('W: %s\nb: %s' % (self.W, self.b))

# Network
class Network(object):
  def __init__(self, layers, lossfunc = 'CrossEntropyLoss'):
    assert(len(layers) == 3) # special model: 3 layers
    if lossfunc == 'CrossEntropyLoss':
      self.lossfunc = CrossEntropyLoss()
    elif lossfunc == 'SquareLoss':
      self.lossfunc = SquareLoss()
    else:
      debug('Unkown loss function: %s' % (lossfunc))
      sys.exit(1)
    # special model building
    self.layers = []
    self.layers.append(FullConnectedLayer(layers[0], layers[1], 'TanhActivator'))
    self.layers.append(FullConnectedLayer(layers[1], layers[2], 'SoftmaxActivator'))

  def predict(self, feature):
    output = feature
    for layer in self.layers:
      layer.forward(output)
      output = layer.output
    return output

  def calc_grad(self, label):
    """
    # Error method 'cause the output layer is SoftmaxActivator
    # specially calc the delta of output layer
    delta = self.layers[-1].activator.backward(
      self.layers[-1].output
    ) * self.lossfunc.backward(self.layers[-1].output, label)
    """
    delta = self.layers[-1].output - label
    for layer in self.layers[-1::-1]:
      layer.backward(delta)
      delta = layer.predelta
    return delta

  def update_weight(self, rate):
    for layer in self.layers:
      layer.update(rate)

  def train_one_sample(self, label, feature, rate):
    self.predict(feature)
    self.calc_grad(label)
    self.update_weight(rate)

  def train(self, labels, data_set, rate, iteration):
    for i in range(iteration):
      for d in range(len(data_set)):
        self.train_one_sample(labels[d], data_set[d], rate)

  def dump(self):
    for layer in self.layers:
      layer.dump()

  def grad_check(self, sample_label, sample_feature):
    self.predict(sample_feature)
    self.calc_grad(sample_label)
    eps = 1e-5
    for layer in self.layers:
      for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
          layer.W[i, j] += eps
          output = self.predict(sample_feature)
          errorpls = self.lossfunc.forward(output, sample_label)

          layer.W[i, j] -= 2 * eps
          output = self.predict(sample_feature)
          errorsub = self.lossfunc.forward(output, sample_label)

          layer.W[i, j] += eps # restore
          expected_grad = (errorpls - errorsub) / (2 * eps)
          debug('Weight[%d, %d]: expect %f - actual %f' % (
            i, j, expected_grad, layer.W_grad[i, j]
          ))
          if abs(expected_grad - layer.W_grad[i, j]) > 1e-3:
            print('<----- grad_check failed. ----->')
            sys.exit(1)

def grad_check_test():
  net = Network([3, 10, 2])
  sample_feature = np.array([[0.38], [0.998], [0.1]])
  sample_label = np.array([[0.1], [0.9]])
  net.grad_check(sample_label, sample_feature)
  print('<----- grad_check_test passed. ----->')

def get_result(vec):
  assert(len(vec) == 10)
  mxx = 0.0
  mxx_index = -1
  for i in range(10):
    # real_val = -np.log(vec[i])
    real_val = vec[i]
    if real_val > mxx:
      mxx = vec[i]
      mxx_index = i
  return mxx_index

def evaluate(network, test_labels, test_data):
  error = 0
  total = len(test_labels)
  for i in range(total):
    label = get_result(test_labels[i])
    predict = get_result(network.predict(test_data[i]))
    if label != predict:
      error += 1
  return float(error) / total

def train_and_evaluate():
  last_error_ratio = 1.0
  epoch = 0

  # parameters of network & training
  rate = 0.0001
  hidden_layer_size = 300
  iterationtime = 2

  network = Network([784, hidden_layer_size, 10])

  (raw_train_features, raw_train_labels), (raw_test_features, raw_test_labels) = keras.datasets.mnist.load_data()
  raw_train_labels = keras.utils.to_categorical(raw_train_labels, num_classes = 10)
  raw_test_labels = keras.utils.to_categorical(raw_test_labels, num_classes = 10)
  debug('Train data shape: ', raw_train_features.shape)
  train_size = len(raw_train_labels)
  test_size = len(raw_test_features)
  train_labels = []
  train_features = []
  test_labels = []
  test_features = []
  for i in range(train_size):
    train_features.append(raw_train_features[i].reshape(784, 1).astype(np.float32))
    train_labels.append(raw_train_labels[i].reshape((10, 1)).astype(np.float32))
  for i in range(test_size):
    test_features.append(raw_test_features[i].reshape(784, 1).astype(np.float32))
    test_labels.append(raw_test_labels[i].reshape((10, 1)).astype(np.float32))

  while True:
    epoch += 1
    network.train(train_labels, train_features, rate, iterationtime)
    print('When %s :> epoch %d finished' % (datetime.datetime.now(), epoch))
    if epoch % 5 == 0:
      error_ratio = evaluate(network, test_labels, test_features)
      print('Epoch %s finished, error ratio: %f' % (epoch, error_ratio))
      if error_ratio > last_error_ratio:
        break
      else:
        last_error_ratio = error_ratio
  print('<----- The final error ratio: %f ----->' % (last_error_ratio))

if __name__ == '__main__':
  grad_check_test()
  train_and_evaluate()