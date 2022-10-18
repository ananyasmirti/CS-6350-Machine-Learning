from math import sqrt
from random import randrange
import matplotlib.pyplot as plt
import numpy as np
    
def batch( dataset):
    th = 1e-6
    lr = 1
    w = []
    cost_fv=[]
    for i in range(len(dataset[0]) - 1):
        w.append(0)
    norm_d = 1
    cost_fv.append(cost_function(w, dataset))
    while norm_d > th:
        grad_v = []
        for i in range(len(w)):
            s = 0
            for j in range(len(dataset)):
                s -= (dataset[j][-1] - inner_prod(w, dataset[j])) * dataset[j][i]
            grad_v.append(s/len(dataset))
        next_w = []
        for i in range(len(w)):
            next_w.append(w[i] - lr * grad_v[i])
        diff = []
        for i in range(len(w)):
            diff.append(next_w[i] - w[i])
        norm_d = norm(diff)
        w = next_w

        cost_fv.append(cost_function(w, dataset))
        lr *= .5
    return w,lr,cost_fv
    
def stochastic( dataset):
    th = 1e-6
    lr = 0.1
    w = []
    cost_fv=[]
    for i in range(len(dataset[0]) - 1):
        w.append(0)

    norm_d = 1
    cost_fv.append(cost_function(w, dataset))

    while norm_d > th:
        grad_v = []
        for i in range(len(w)):
            s = 0
            rr = randrange(0, len(dataset))
            s -= (dataset[rr][-1] - inner_prod(w, dataset[rr])) * dataset[rr][i]
            grad_v.append(s)
        next_w = []

        for i in range(len(w)):
            next_w.append(w[i] - lr * grad_v[i])
        
        diff = []
        for i in range(len(w)):
            diff.append(next_w[i] - w[i])
        norm_d = norm(diff)

        w = next_w

        cost_fv.append(cost_function(w, dataset))
        lr *= .999
    return w,r,cost_fv


def cost_function(w, dataset):
    er = 0
    for r in dataset:
        er += .5 * (r[-1] - inner_prod(w, r)) ** 2
    return er

def inner_prod(v1, v2):
    pr = 0
    for i in range(len(v1)):
        pr += v1[i] * v2[i]
    return pr

def norm(v):
    ss = 0
    for c in v:
        ss += c ** 2
    return sqrt(ss)

def extract_data_from_csv(csv_file_path):
    dataset = []
    with open(csv_file_path, 'r') as f:
        for line in f:
            dataset.append(line.strip().split(','))
    return dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_s = extract_data_from_csv('./concrete/train.csv')
test_s = extract_data_from_csv('./concrete/test.csv')

train= []
for r in train_s:
    f= []
    for x in r:
        f.append(float(x))
    train.append(f)

test= []
for r in test_s:
    f = []
    for x in r:
        f.append(float(x))
    test.append(f)


w,r,cost_fv = batch(dataset=train)
test_e = cost_function(w, test)
print('BGD')
print('Cost on testing data - ', test_e)
print('Start LR value 1')
print('LR value per step 0.5')
print(f'Final weights: {w}')
print(f'Final Lr: {r}')

plt.plot(cost_fv)
plt.ylabel('Cost Function Value')
plt.xlabel('Step')
plt.title('BGD Cost Function')
plt.savefig('BGD.png')



w,r,cost_fv = stochastic(train)
test_e = cost_function(w, test)
print('SGD')
print('Cost on testing data- ', test_e)
print('Start LR value 0.1')
print('LR value per step 0.999')
print(f'Final weights: {w}')
print(f'Final Lr: {r}')

plt.plot(cost_fv)
plt.ylabel('Value of Cost Function ')
plt.xlabel('Step')
plt.title('SGD Cost Function')
plt.savefig('SGD.png')

X = np.transpose(np.array([row[:-1] for row in train]))
Y = np.array([row[-1] for row in train])

w_a= np.dot(np.linalg.inv(np.dot(X, np.transpose(X))), np.dot(X, Y))

print(f'The analytical solution to the optimal weight vector is: \n{w_a}')