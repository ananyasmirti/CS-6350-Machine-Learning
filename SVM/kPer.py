import pandas as pd
import numpy as np
    
def Perceptron_k( x, y,gamma):
    num_s = x.shape[0]
    id= np.arange(num_s)
    a = np.array([x for x in range(num_s)])
    a = np.reshape(a, (-1, 1))
    y = np.reshape(y, (-1, 1))
    k = kernel_gauss(x,x, gamma)
    for t in range(100):
        np.random.shuffle(id)
        for i in range(num_s):
            ay = np.multiply(a, y)
            ayk = np.matmul(k[id[i], :], ay)
            if ayk * y[id[i]] <= 0:
                a[id[i]] = a[id[i]] + 1
    return a

def Perceptron_k_predict(a, x0, y0, x,gamma):
    k = kernel_gauss(x0, x, gamma)
    ay = np.reshape(np.multiply(a, np.reshape(y0, (-1, 1))), (1, -1))
    y = np.matmul(ay, k)
    y = np.reshape(y, (-1,1))
    y[y > 0] = 1
    y[y <=0] = -1
    return y

def kernel_gauss( x1, x2, gamma):
    a1 = np.tile(x1, (1, x2.shape[0]))
    a1 = np.reshape(a1, (-1,x1.shape[1]))
    a2 = np.tile(x2, (x1.shape[0], 1))
    k = np.exp(np.sum(np.square(a1 - a2),axis=1) / -gamma)
    k = np.reshape(k, (x1.shape[0], x2.shape[0]))
    return k

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

gamma = np.array([ 0.1, 0.5, 1, 5,100])

for g in gamma:
    print('For Gamma: ', g)
    p = Perceptron_k(train_x, train_y,g)
    y = Perceptron_k_predict(p, train_x, train_y, train_x,g)
    err_tr = np.sum(np.abs(y - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]
    y = Perceptron_k_predict(p, train_x, train_y, test_x,g)
    err_te = np.sum(np.abs(y - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
    print('Kernel Perceptron')
    print('Train error: ', err_tr)
    print('Test error: ', err_te)