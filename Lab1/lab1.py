'''
Lab 1
This code implements a two-layer perceptron with
the backpropagation algorithm to solve the parity problem.

Yue Lin (lin.3326 at osu.edu)
Created: 9/22/2020
'''

from random import seed, randrange, uniform, shuffle
from math import exp
import csv


# Initialize network
def init_net(n_inputs, n_hidden, n_outputs):
  net = list()
  hidden_layer = [{'w':[uniform(-1, 1) for i in range(n_inputs + 1)]}
                  for i in range(n_hidden)]
  net.append(hidden_layer)
  output_layer = [{'w':[uniform(-1, 1) for i in range(n_hidden + 1)]}
                  for i in range(n_outputs)]
  net.append(output_layer)
  return net


# Define activation function
def transfer(w, inputs):
  # activation function: logistic sigmoid with a = 1
  v = w[-1]
  for i in range(len(w)-1):
    v += w[i] * inputs[i]
  phi = 1. / (1. + exp(-v))
  return phi


# Forward propagate
def fwd_prop(net, row):
  inputs = row
  for layer in net:
    new_inputs = []
    for neuron in layer:
      neuron['y'] = transfer(neuron['w'], inputs)
      new_inputs.append(neuron['y'])
    inputs = new_inputs
  return inputs


# Backward propagate error
def bwd_prop_err(net, d):
  for i in reversed(range(len(net))):
    layer = net[i]
    errors = list()
    if i != len(net)-1:
      for j in range(len(layer)):
        error = 0.
        for neuron in net[i + 1]:
          error += (neuron['w'][j] * neuron['delta'])
        errors.append(error)
    else:
      for j in range(len(layer)):
        neuron = layer[j]
        errors.append(d[j] - neuron['y'])
    
    for j in range(len(layer)):
      neuron = layer[j]
      phi_d = neuron['y'] * (1. - neuron['y'])
      neuron['delta'] = errors[j] * phi_d


# Update weights
def update_weights(net, row, lr):
  for i in range(len(net)):
    inputs = row[:-1]
    if i != 0:
      inputs = [neuron['y'] for neuron in net[i - 1]]
    for neuron in net[i]:
      for j in range(len(inputs)):
        neuron['w'][j] += lr * neuron['delta'] * inputs[j]
      neuron['w'][-1] += lr * neuron['delta']


def train(net, data, lr, n_outputs):
  epoch = 0
  c = True
  
  # Train
  while c:
    epoch += 1
    sum_error = 0
    max_abs_error = 0
    c = False
    
    # Randomization
    shuffle(data)

    # Online learning
    for row in data:
      outputs = fwd_prop(net, row)
      d = row[-n_outputs:]
      bwd_prop_err(net, d)
      update_weights(net, row, lr)
      
      # Stopping criteria
      sum_error += sum([(d[i]-outputs[i])**2 for i in range(n_outputs)])
      if sum([abs(d[i]-outputs[i]) for i in range(n_outputs)]) > max_abs_error:
        max_abs_error = sum([abs(d[i]-outputs[i]) for i in range(n_outputs)])
      if max_abs_error > 0.05:
        c = True
    print('>epoch=%d, lrate=%.3f, sum_error=%.3f, max_abs_error=%.3f'
          % (epoch, lr, sum_error, max_abs_error))


# Wrapper
n_inputs = 4  # Number of input neurons
n_hidden = 4  # Number of hidden neurons
n_outputs = 1 # Number of output neurons
lr = 0.5

# Read training data
with open("lab1-train.csv", "r") as f:
  data = []
  for line in f:
      data_line = line.rstrip().split(",")
      data_line = [int(i) for i in data_line] 
      data.append(data_line)
print(data)

# Initialize network
net = init_net(n_inputs, n_hidden, n_outputs)

# Train
train(net, data, lr, n_outputs)
for layer in net:
  print(layer)
