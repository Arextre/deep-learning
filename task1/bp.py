#!/usr/bin/env python3

import numpy as np
import sys
import keras
import datetime

ISDEBUG = False
def debug(*args, **kwargs):
  if ISDEBUG:
    print(*args, **kwargs)
def RaiseNotImplementedError():
  raise NotImplementedError('Not implemented yet.')

class ReLUActivator(object):
  def forward(self, input):
    return np.maximum(0, input)
  def backward(self, output):
    return np.array([1 if x > 0 else 0 for x in output]).reshape(output.shape)

class SigmoidActivator(object):
  def forward(self, input):
    return 1.0 / (1.0 + np.exp(-input))
  def backward(self, output):
    return output * (1 - output)

class TanhActivator(object):
  def forward(self, input):
    return np.tanh(input)
  def backward(self, output):
    return 1 - output**2

class LeakyReLUActivator(object):
  def forward(self, input):
    return np.maximum(0.1 * input, input)
  def backward(self, output):
    return np.array([1 if x > 0 else 0.1 for x in output]).reshape(output.shape)

class SoftmaxActivator(object):
  def forward(self, input):
    return np.exp(input) / np.sum(np.exp(input))
  def backward(self, output):
    # wrong backward propagation implementation, 
    # maybe it cannot be wrapped by a single function
    # return output * (1 - output)
    RaiseNotImplementedError()

class Network(object):
  def __init__(self, layers):
    assert(len(layers) == 3) # special model required
    self.n = layers[0]
    self.m = layers[1]
    self.k = layers[2]
    self.W1 = np.random.uniform(-0.1, 0.1, (self.m, self.n))
    self.W2 = np.random.uniform(-0.1, 0.1, (self.k, self.m))
    self.b1 = np.zeros((self.m, 1))
    self.b2 = np.zeros((self.k, 1))
    self.f1 = ReLUActivator()
    self.f2 = SoftmaxActivator()
  
  def predict(self, feature):
    assert(len(feature) == self.n)
    self.o0 = feature
    self.z1 = np.dot(self.W1, self.o0) + self.b1
    self.o1 = self.f1.forward(self.z1)
    self.z2 = np.dot(self.W2, self.o1) + self.b2
    self.o2 = self.f2.forward(self.z2)
    self.output = self.o2
    return self.output
  
  def backward(self, label):
    assert(len(label) == self.k)
    self.delta2 = self.o2 * np.sum(label) - label
    self.delta1 = self.f1.backward(self.o1) * np.dot(self.W2.T, self.delta2)
    self.W2_grad = np.dot(self.delta2, self.o1.T)
    self.b2_grad = self.delta2
    self.W1_grad = np.dot(self.delta1, self.o0.T)
    self.b1_grad = self.delta1
  
  def update(self, rate):
    self.W1 += -rate * self.W1_grad
    self.b1 += -rate * self.b1_grad
    self.W2 += -rate * self.W2_grad
    self.b2 += -rate * self.b2_grad

  def train(self, labels, features, rate, iteration):
    for i in range(iteration):
      for d in range(len(labels)):
        self.predict(features[d])
        self.backward(labels[d])
        self.update(rate)

  def loss(self, label):
    assert(self.k == len(label))
    return -np.sum(label * np.log(self.output))

def grad_check():
  net = Network([3, 8, 2])
  feature = np.array([[1], [2], [3]])
  label = np.array([[0.8], [0.1]])
  net.predict(feature)
  net.backward(label)
  eps = 1e-5
  for i in range(net.W1.shape[0]):
    for j in range(net.W1.shape[1]):
      net.W1[i, j] += eps
      net.predict(feature)
      errorpls = net.loss(label)

      net.W1[i, j] -= eps * 2
      net.predict(feature)
      errorsub = net.loss(label)

      net.W1[i, j] += eps
      expected_grad = (errorpls - errorsub) / (2 * eps)
      actual_grad = net.W1_grad[i, j]
      debug('expect: %f - actual %f' % (expected_grad, actual_grad))
      if abs(expected_grad - actual_grad) > 1e-3:
        print('<----- Grad_check failed. ----->')
        sys.exit(1)
  
  for i in range(net.W2.shape[0]):
    for j in range(net.W2.shape[1]):
      net.W2[i, j] += eps
      net.predict(feature)
      errorpls = net.loss(label)

      net.W2[i, j] -= eps * 2
      net.predict(feature)
      errorsub = net.loss(label)

      net.W2[i, j] += eps
      expected_grad = (errorpls - errorsub) / (2 * eps)
      actual_grad = net.W2_grad[i, j]
      debug('<----- expect: %f - actual %f ----->' % (expected_grad, actual_grad))
      if abs(expected_grad - actual_grad) > 1e-3:
        print('<----- Grad_check failed. ----->')
        sys.exit(1)

if __name__ == '__main__':
  grad_check()
