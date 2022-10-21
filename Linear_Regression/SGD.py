import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

def cost_fucntion(w, dataset):
    sum = 0
    for i in range(len(dataset.index)):
        x_i = np.array([dataset.iloc[i, 0], dataset.iloc[i, 1], dataset.iloc[i, 2],
                       dataset.iloc[i, 3], dataset.iloc[i, 4], dataset.iloc[i, 5],  dataset.iloc[i, 6]])
        wt_xi = np.matmul(np.transpose(w), x_i)
        s = math.pow(dataset.iloc[i, 7] - wt_xi, 2)
        sum += s

    loss = sum / 2
    return loss


def gradient(w, dataset):
    grad = np.array([0] * 7, dtype=float)

    x = np.array(dataset[0:7])
    wt = np.matmul(np.transpose(w), x)
    y = dataset[7] - wt
    grad= y * x
    return grad

def sgd(start, lr, n_iter, dataset):
    w = start
    diff=1
    w_new=start
    l = cost_fucntion(w, dataset)
    lastloss=0
    cost_v=[]
    cost_v.append(l)
    for ep in range(n_iter):
      if diff <= 1e-6: break
      for i in range(len(dataset)):
          rr = randrange(0, len(dataset))
          d = gradient(w, dataset.iloc[rr].values)
          w_new = w + lr * d
          diff = []
          loss = cost_fucntion(w_new, dataset)
          cost_v.append(loss)
          w=w_new
          diff = abs(loss - lastloss)
          lastloss = loss
    return w, cost_v


train_data = pd.read_csv('./concrete/train.csv')
test_data = pd.read_csv('./concrete/test.csv')
initial_w = np.array([0] * 7, dtype=float)

w, cost_function = sgd(start=initial_w, lr=0.01, n_iter=100, dataset=train_data)
print("Weight:", w)
print("Learning rate:", 0.01)
print("Cost function of test data:", cost_fucntion(w, test_data))



w, cost_function1 = sgd(start=initial_w, lr=0.01/2, n_iter=100, dataset=train_data)
print("Weight:", w)
print("Learning rate:", 0.01/2)
print("Cost function of test data:", cost_fucntion(w, test_data))


w, cost_function2 = sgd(start=initial_w, lr=0.01/4, n_iter=100, dataset=train_data)
print("Weight:", w)
print("Learning rate:", 0.01/4)
print("Cost function of test data:", cost_fucntion(w, test_data))

w, cost_function3 = sgd(start=initial_w, lr=0.01/8, n_iter=100, dataset=train_data)
print("Weight:", w)
print("Learning rate:", 0.01/8)
print("Cost function of test data:", cost_fucntion(w, test_data))


plt.plot(cost_function,label="lr=0.01")
plt.plot(cost_function1,label="lr=0.005")
plt.plot(cost_function2,label="lr=0.0025")
plt.plot(cost_function3,label="lr=0.00125")
plt.ylabel('Cost Function Value')
plt.xlabel('Step')
plt.title('SGD Cost Function')
plt.legend()
plt.savefig('SGD.png')
