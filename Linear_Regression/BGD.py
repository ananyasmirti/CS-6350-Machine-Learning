

from math import sqrt
from random import randrange
import matplotlib.pyplot as plt
import numpy as np
    
def batch(lr, dataset):
    th = 1e-6
    
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
        
    return w,lr,cost_fv

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


w,r,cost_fv = batch(lr=0.5,dataset=train)
test_e = cost_function(w, test)

print('BGD')
print('Cost on testing data - ', test_e)
print(f'weights: {w}')
print(f'Lr: {r}')

w1,r1,cost_fv1 = batch(lr=0.25,dataset=train)
test_e = cost_function(w, test)

print('BGD')
print('Cost on testing data - ', test_e)
print(f'weights: {w1}')
print(f'Lr: {r1}')

w2,r2,cost_fv2 = batch(lr=0.25/2,dataset=train)
test_e = cost_function(w1, test)

print('BGD')
print('Cost on testing data - ', test_e)
print(f'weights: {w2}')
print(f'Lr: {r2}')

w3,r3,cost_fv3 = batch(lr=0.25/4,dataset=train)
test_e = cost_function(w3, test)

print('BGD')
print('Cost on testing data - ', test_e)
print(f'weights: {w3}')
print(f'Lr: {r3}')

w4,r4,cost_fv4 = batch(lr=0.03125,dataset=train)
test_e = cost_function(w4, test)

print('BGD')
print('Cost on testing data - ', test_e)
print(f'weights: {w4}')
print(f'Lr: {r4}')


plt.plot(cost_fv,label="lr=0.5")
plt.plot(cost_fv1,label="lr=0.25")
plt.plot(cost_fv2,label="lr=0.125")
plt.plot(cost_fv3,label="lr=0.0625")
plt.plot(cost_fv4,label="lr=0.03125")

plt.ylabel('Cost Function Value')
plt.xlabel('Step')
plt.title('BGD Cost Function')
plt.legend()
plt.savefig('BGD.png')

X = np.transpose(np.array([row[:-1] for row in train]))
Y = np.array([row[-1] for row in train])

w_a= np.dot(np.linalg.inv(np.dot(X, np.transpose(X))), np.dot(X, Y))

print(f'The analytical solution to the optimal weight vector is: \n{w_a}')