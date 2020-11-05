from libsvm.svmutil import *
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt


# Read training and test data
def get_data(filename):
  data = load_svmlight_file(filename)
  return data


# Classification using linear SVMs
def linear_svm(train_data, c, test_data):
  # Get data
  train_x = train_data[0]
  train_y = train_data[1]
  test_x = test_data[0]
  test_y = test_data[1]

  # Train svm
  prob = svm_problem(train_y, train_x)
  param_str = '-t 0 -c 2e' + str(c)
  print("Param: " + param_str)
  param = svm_parameter(param_str)
  m = svm_train(prob, param)

  # Test
  p_label, p_acc, p_val = svm_predict(test_y, test_x, m)
  return p_acc


# Wrapper
train_file = "ncRNA_s.train.txt"
test_file = "ncRNA_s.test.txt"

train_data = get_data(train_file)
test_data = get_data(test_file)

acc = []
for c in range(-4, 9):
  p_acc = linear_svm(train_data, c, test_data)
  acc.append(p_acc[0])

cost = ['2e-4', '2e-3', '2e-2', '2e-1', '2e0', '2e1', '2e2', '2e3', '2e4', '2e5', '2e6', '2e7', '2e8']
plt.plot(cost, acc)
plt.xlabel('Cost')
plt.ylabel('Accuracy')