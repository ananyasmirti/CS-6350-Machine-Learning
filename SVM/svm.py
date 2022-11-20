import pandas as pd
import numpy as np
import scipy.optimize as opt

        
def primal_SVM(d,lr,C, x, y):
    num_s = x.shape[0]
    s = x.shape[1]
    wei  = np.zeros(s)
    i = np.arange(num_s)
    for t in range(100):
        np.random.shuffle(i)
        x = x[i,:]
        y = y[i]
        for j in range(num_s):
            tmp = y[j] * np.sum(np.multiply(wei, x[j,:]))
            a = np.copy(wei)
            a[s-1] = 0
            if tmp <= 1:
                a = a - C * num_s * y[j] * x[j,:]
            lr = lr / (1 + lr / d * t)
            wei = wei - lr * a
            # print(wei)

    return wei

def primal_SVM1(lr,C, x, y):
    num_s = x.shape[0]
    di = x.shape[1]
    wei = np.zeros(di)
    i = np.arange(num_s)
    for t in range(100):
        np.random.shuffle(i)
        x = x[i,:]
        y = y[i]
        for j in range(num_s):
            tmp = y[j] * np.sum(np.multiply(wei, x[j,:]))
            a = np.copy(wei)
            a[di-1] = 0
            if tmp <= 1:
                a = a - C * num_s * y[j] * x[j,:]
            lr = lr / (1 + t)
            wei = wei - lr * a
            # print(wei)

    return wei
    
def object(al, x, y):
    le = 0
    le = le - np.sum(al)
    ayx = np.multiply(np.multiply(np.reshape(al,(-1,1)), np.reshape(y, (-1,1))), x)
    le = le + 0.5 * np.sum(np.matmul(ayx, np.transpose(ayx)))
    return le

def convert( al,y):
    t1 = np.matmul(np.reshape(al,(1, -1)), np.reshape(y,(-1,1)))
    return t1[0]

def dual_SVM(C,x, y):
    num_s = x.shape[0]
    bands = [(0, C)] * num_s
    cons = ({'type': 'eq', 'fun': lambda al: convert(al, y)})
    al0 = np.zeros(num_s)
    r = opt.minimize(lambda al: object(al, x, y), al0,  method='SLSQP', bounds=bands,constraints=cons, options={'disp': False})
    wei = np.sum(np.multiply(np.multiply(np.reshape(r.x,(-1,1)), np.reshape(y, (-1,1))), x), axis=0)
    id = np.where((r.x > 0) & (r.x < C))
    b =  np.mean(y[id] - np.matmul(x[id,:], np.reshape(wei, (-1,1))))
    wei = wei.tolist()
    wei.append(b)
    wei = np.array(wei)
    return wei

def kernel_gauss(x1, x2, gamma):
    a = np.tile(x1, (1, x2.shape[0]))
    a = np.reshape(a, (-1,x1.shape[1]))
    b = np.tile(x2, (x1.shape[0], 1))
    d = np.exp(np.sum(np.square(a - b),axis=1) / -gamma)
    d = np.reshape(d, (x1.shape[0], x2.shape[0]))
    return d

def object_guass( al, k, y):
    l = 0
    l = l - np.sum(al)
    ay = np.multiply(np.reshape(al,(-1,1)), np.reshape(y, (-1,1)))
    ayay = np.matmul(ay, np.transpose(ay))
    l = l + 0.5 * np.sum(np.multiply(ayay, k))
    return l
    
def gauss_SVM(C,gamma, x, y):
    num_s = x.shape[0]
    bands = [(0, C)] * num_s
    cons = ({'type': 'eq', 'fun': lambda al: convert(al, y)})
    al0 = np.zeros(num_s)
    d = kernel_gauss(x, x, gamma)
    r = opt.minimize(lambda al: object_guass(al, d, y), al0,  method='SLSQP', bounds=bands,constraints=cons, options={'disp': False})
    return r.x
    
def gauss_predict(gamma, al, x0, y0, x):
    k = kernel_gauss(x0, x, gamma)
    k = np.multiply(np.reshape(y0, (-1,1)), k)
    y = np.sum(np.multiply(np.reshape(al, (-1,1)), k), axis=0)
    y = np.reshape(y, (-1,1))
    y[y > 0] = 1
    y[y <=0] = -1
    return y

train = pd.read_csv('./bank-note/train.csv', header=None)

r = train.values
col = r.shape[1]
row = r.shape[0]
train_x = np.copy(r)
train_x[:,col - 1] = 1
train_y = r[:, col - 1]
train_y = 2 * train_y - 1

test = pd.read_csv('./bank-note/test.csv', header=None)
r = test.values
col = r.shape[1]
row = r.shape[0]
test_x = np.copy(r)
test_x[:,col - 1] = 1
test_y = r[:, col - 1]
test_y = 2 * test_y - 1

C_list = np.array([100, 500, 700])
C_list = C_list/ 873
gamma_list = np.array([ 0.1, 0.5, 1, 5, 100])

for C in C_list:
    print('C: ', C)
   
    wei = primal_SVM(0.1,0.1,C,train_x, train_y)
    wei = np.reshape(wei, (5,1))

    pred = np.matmul(train_x, wei)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1
    err_train = np.sum(np.abs(pred - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]

    pred = np.matmul(test_x, wei)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1

    err_test = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
    print('Primal SVM Linear')
    print('Train error: ', err_train)
    print('Test error: ', err_test)
    wei = np.reshape(wei, (1,-1))
    # print('weight:', wei)

    wei = primal_SVM1(0.1,C,train_x, train_y)
    wei = np.reshape(wei, (5,1))

    pred = np.matmul(train_x, wei)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1
    err_train = np.sum(np.abs(pred - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]

    pred = np.matmul(test_x, wei)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1

    err_test = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
    print('Primal SVM Linear with different LR formula')
    print('Train error: ', err_train)
    print('Test error: ', err_test)
    wei = np.reshape(wei, (1,-1))

    wei = dual_SVM(C,train_x[:,[x for x in range(col - 1)]] ,train_y)
    # print('weight: ', wei)

    wei = np.reshape(wei, (5,1))

    pred = np.matmul(train_x, wei)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1
    err_train = np.sum(np.abs(pred - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]

    pred = np.matmul(test_x, wei)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1

    err_test= np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
    print('Dual SVM Linear ')
    print('Train error: ', err_train)
    print('Test error: ', err_test)


    c = 0
    for gamma in gamma_list:
        print('Gamma: ', gamma)
        al = gauss_SVM(C,gamma,train_x[:,[x for x in range(col - 1)]] ,train_y)
        id = np.where(al > 0)[0]
        print('Support Vectors: ', len(id))
    
        y = gauss_predict(gamma,al, train_x[:,[x for x in range(col - 1)]], train_y, train_x[:,[x for x in range(col - 1)]])
        err_train = np.sum(np.abs(y - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]

     
        y = gauss_predict(gamma,al,train_x[:,[x for x in range(col - 1)]], train_y, test_x[:,[x for x in range(col - 1)]])
        err_test = np.sum(np.abs(y - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
        print('Gaussian SVM non-Linear ')
        print('Train error: ', err_train)
        print('Test error: ', err_test)
        
        if(C == 500/873):
            if c > 0:
                intersect = len(np.intersect1d(id, old_id))
                print('Intersections: ', intersect)
            c = c + 1
            old_id = id