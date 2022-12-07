import numpy as np
import pandas as pd

    
def MAP(x, y,v,r0,d):
    n_s = x.shape[0]
    di = x.shape[1]
    wei = np.zeros([1, di])
    id = np.arange(n_s)
    for t in range(100):
        np.random.shuffle(id)
        x = x[id,:]
        y = y[id]
        for i in range(n_s):
            x_i = x[i,:].reshape([1, -1])
            te = y[i] * np.sum(np.multiply(wei, x_i))
            g = - (n_s * y[i] * x_i )/ (1 + np.exp(te)) + wei / v
            #print(g)
            r = r0 / (1 + r0 / d * t)
            #r0=r0/2
            wei = wei - (r * g)
            #print(wei)
    return wei.reshape([-1,1])

def ML( x, y,r0,d):
    n_s = x.shape[0]
    di = x.shape[1]
    wei = np.zeros([1, di])
    id = np.arange(n_s)
    for t in range(100):
        np.random.shuffle(id)
        x = x[id,:]
        y = y[id]
        for i in range(n_s):
            te = y[i] * np.sum(np.multiply(wei, x[i,:]))
            g = - n_s * y[i] * x[i,:] / (1 + np.exp(te))
            r = r0 / (1 + r0 / d * t)
            wei = wei - r * g
    return wei.reshape([-1,1])


train = pd.read_csv('./bank-note/train.csv', header=None)
r = train.values
n_col = r.shape[1]
train_x = np.copy(r)
train_x[:,n_col - 1] = 1
train_y = r[:, n_col - 1]
train_y = 2 * train_y - 1

test = pd.read_csv('./bank-note/test.csv', header=None)
r = test.values
n_col = r.shape[1]
test_x = np.copy(r)
test_x[:,n_col - 1] = 1
test_y = r[:, n_col - 1]
test_y = 2 * test_y - 1

# train_x = np.array([[0.5, -1, -3, 1], [-1,-2,-2,1], [1.5, 0.2, -2.5, 1]])
# train_y = np.array([1,-1,1])

# wei= MAP(train_x, train_y,1,0.01,0.1)
v_l = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

print('MAP Results:')
for v in v_l:
    
    print('V:', v)
    wei= MAP(train_x, train_y,v,0.001,0.1)
    pred = np.matmul(train_x, wei)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1
    e_train = np.sum(np.abs(pred - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]

    pred = np.matmul(test_x, wei)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1

    e_test = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
    print('Train error: ', e_train)
    print('Test error: ',e_test)


print('ML Results:')
wei= ML(train_x, train_y,0.001,0.1)

pred = np.matmul(train_x, wei)
pred[pred > 0] = 1
pred[pred <= 0] = -1
e_train = np.sum(np.abs(pred - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]

pred = np.matmul(test_x, wei)
pred[pred > 0] = 1
pred[pred <= 0] = -1

e_test = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
print('Train error: ', e_train)
print('Test error: ',e_test)