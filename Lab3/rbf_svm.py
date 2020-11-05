from libsvm.svmutil import *
from sklearn.datasets import load_svmlight_file
import random
import matplotlib.pyplot as plt


# Divide datasets
def divide_data(filename, n, m):
  # Read the entire training dataset
  with open(filename, 'r') as f:
    lines = f.read().splitlines()
    f.close()
  
  # Divide
  lines = random.sample(lines, int(len(lines) * 0.5))
  _cv = [lines[i::n] for i in range(n)]
  _train = _cv[:m] + _cv[m + 1:]
  _train = [x for sublist in _train for x in sublist]
  _val = _cv[m]
  # print(_train)
  # print(_val)

  # Get y values
  train_y = [int(line.split(' ')[0]) for line in _train]
  val_y = [int(line.split(' ')[0]) for line in _val]
  # print(train_y)
  # print(val_y)

  # Get x values
  train_x = [line.split(' ')[1:] for line in _train]
  for i in range(len(train_x)):
    line = train_x[i]
    for j in range(len(line)):
      train_x[i][j] = float(train_x[i][j].split(':')[1])
  val_x = [line.split(' ')[1:] for line in _val]
  for i in range(len(val_x)):
    line = val_x[i]
    for j in range(len(line)):
      val_x[i][j] = float(val_x[i][j].split(':')[1])
  # print(train_x)
  # print(val_x)

  return train_x, train_y, val_x, val_y


# Select parameters for RBF kernel SVMs
def rbf_svm_param(filename, c, alpha):
  acc = 0

  for i in range(5):
    # Prepare data
    train_x, train_y, val_x, val_y = divide_data(filename, 5, i)

    # Train svm
    prob = svm_problem(train_y, train_x)
    param_str = '-t 2 -c 2e' + str(c) + ' -g 2e' + str(alpha)
    print("Param: " + param_str)
    param = svm_parameter(param_str)
    m = svm_train(prob, param)

    # Test
    p_label, p_acc, p_val = svm_predict(val_y, val_x, m)
    acc += p_acc[0]

  return acc / 5


# Read training and test data
def get_data(filename):
  data = load_svmlight_file(filename)
  return data


# Classification using RBF kernel SVMs
def rbf_svm(train_data, c, alpha, test_data):
  # Get data
  train_x = train_data[0]
  train_y = train_data[1]
  test_x = test_data[0]
  test_y = test_data[1]

  # Train svm
  prob = svm_problem(train_y, train_x)
  param_str = '-t 2 -c 2e' + str(c) + ' -g 2e' + str(alpha)
  print("Param: " + param_str)
  param = svm_parameter(param_str)
  m = svm_train(prob, param)

  # Test
  p_label, p_acc, p_val = svm_predict(test_y, test_x, m)
  return p_acc


# Wrapper
# Select parameters
train_file = "ncRNA_s.train.txt"
acc = [[0] * 13 for i in range(13)]
for c in range(-4, 9):
  for alpha in range(-4, 9):
    p_acc = rbf_svm_param(train_file, c, alpha)
    acc[int(c+4)][int(alpha+4)] = p_acc
with open('acc_mtx.txt', 'w') as fw:
  fw.write('\n'.join(['\t'.join([str(round(cell,2)) for cell in row]) for row in acc]))


# Read training and test data
train_file = "ncRNA_s.train.txt"
test_file = "ncRNA_s.test.txt"

train_data = get_data(train_file)
test_data = get_data(test_file)

c = 6
alpha = -3
p_acc = rbf_svm(train_data, c, alpha, test_data)
print(p_acc[0])