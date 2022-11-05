import pandas as pd
import numpy as np

def standard(x, y,T,lr):
    ns= x.shape[0]
    dim = x.shape[1]
    wei = np.zeros(dim)
    b = np.arange(ns)
    for t in range(T):
        np.random.shuffle(b)
        x = x[b,:]
        y = y[b]
        for i in range(ns):
            t = np.sum(x[i] * wei)
            if not (t * y[i] > 0):
                wei = wei + lr * y[i] * x[i]
    return wei

def voted(x, y,T, lr):
    ns = x.shape[0]
    m = x.shape[1]
    wei = np.zeros(m)
    b = np.arange(ns)
    list_c = np.array([])
    list_w = np.array([])
    c = 0
    for t in range(T):
        
        x = x[b,:]
        y = y[b]
        for i in range(ns):
            tmp = np.sum(x[i] * wei)
            if not (tmp * y[i] > 0):
                list_w = np.append(list_w, wei)
                list_c = np.append(list_c, c)
                wei = wei + lr * y[i] * x[i]
                c = 1
            else:
                c = c + 1
    n1 = list_c.shape[0]
    list_w = np.reshape(list_w, (n1,-1))
    return list_c, list_w
    
def average(x, y,T,lr):
    ns= x.shape[0]
    m = x.shape[1]
    wei = np.zeros(m)
    a = np.zeros(m)
    b = np.arange(ns)
    for t in range(T):
        
        x = x[b,:]
        y = y[b]
        for i in range(ns):
            t = np.sum(x[i] * wei)
            if not (t * y[i] > 0):
                wei = wei + lr * y[i] * x[i]
            a = a + wei
    return a

train = pd.read_csv('./bank-note/train.csv', header=None)
r = train.values
nc = r.shape[1]
nr = r.shape[0]
train_x = np.copy(r)
train_x[:,nc - 1] = 1
train_y = r[:, nc - 1]
train_y = 2 * train_y - 1

test = pd.read_csv('./bank-note/test.csv', header=None)
r = test.values
nc = r.shape[1]
num_row = r.shape[0]
test_x = np.copy(r)
test_x[:,nc - 1] = 1
test_y = r[:, nc - 1]
test_y = 2 * test_y - 1

print("=================Standard Perceptron==================")
wei = standard(train_x, train_y,10,0.01)
wei = np.reshape(wei, (-1,1))
pred = np.matmul(test_x, wei)
pred[pred > 0] = 1
pred[pred <= 0] = -1
error = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
print('Test Error:', error)
print('Weights:',wei)

print("=================Voted Perceptron==================")
c_list, w_list = voted(train_x, train_y,10,0.01)
c_list = np.reshape(c_list, (-1,1))
print('Weights:')
print(w_list)
w_list = np.transpose(w_list)
pred1 = np.matmul(test_x, w_list)
pred1[pred1 >0] = 1
pred1[pred1 <=0] = -1
voted = np.matmul(pred1, c_list)
voted[voted >0] = 1
voted[voted<=0] = -1
error = np.sum(np.abs(voted - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
print('Count')
print(c_list)
print('Test Error',error)

print("=================Average Perceptron==================")
wei1 = average(train_x, train_y,10,0.01)
wei1 = np.reshape(wei1, (-1,1))
pred = np.matmul(test_x, wei1)
pred[pred > 0] = 1
pred[pred <= 0] = -1
err = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
print('Test Error:', error)
print('Weights:',wei1)