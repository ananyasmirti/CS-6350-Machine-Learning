import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NN():
    def __init__(self, wid):
        self.wid = wid
        self.W1 = np.array([]) 
        self.b1 = np.array([]) 
        self.W2 = np.array([]) 
        self.b2 = np.array([]) 
        self.W3 = np.array([]) 
        self.b3 = np.array([]) 

    def weight(self, d, wi=True):
        self.W1 = np.random.normal(size=(d,self.wid)) if wi else np.zeros((d,self.wid))
        self.b1 = np.random.normal(size=(self.wid)) if wi else np.zeros((self.wid))
        self.W2 = np.random.normal(size=(self.wid,self.wid)) if wi else np.zeros((self.wid,self.wid)) 
        self.b2 = np.random.normal(size=(self.wid)) if wi else np.zeros((self.wid)) 
        self.W3 = np.random.normal(size=(self.wid,1)) if wi else np.zeros((self.wid,1)) 
        self.b3 = np.random.normal(size=(1)) if wi else np.zeros((1)) 

    
    def forward(self, X):
        S1 = np.dot(X, self.W1) + self.b1 
        Z1 = self.sigmoid(S1) 

        S2 = np.dot(Z1, self.W2) + self.b2 
        Z2 = self.sigmoid(S2) 

        scores = np.dot(Z2, self.W3) + self.b3 
        ca = (S1, Z1, S2, Z2)

        return scores, ca

    def backwards(self, X, y, scores, ca):
        S1, Z1, S2, Z2 = ca

        dy = (scores - y).reshape((1,1)) 
        dW3 = np.dot(Z2.reshape((1,-1)).T, dy) 
        db3 = np.sum(dy, axis=0) 
        dZ2 = np.dot(dy, self.W3.reshape((-1,1)).T) 

        dsig_2 = self.sigmoid(S2) * (1 - self.sigmoid(S2)) * dZ2 
        dW2 = np.dot(Z1.reshape((1,-1)).T, dsig_2) 
        db2 = np.sum(dsig_2, axis=0) 
        dZ1 = np.dot(dsig_2, self.W2.T) 

        dsig_1 = self.sigmoid(S1) * (1 - self.sigmoid(S1)) * dZ1
        dW1 = np.dot(X.reshape((1,-1)).T, dsig_1)
        db1 = np.sum(dsig_1, axis=0) 
        
        return dW1, db1, dW2, db2, dW3, db3


    def train(self, X, y, T=100,th=1e-6,r=0.1, wi=True, r_=None):
        m, d = X.shape
        self.weight(d, wi=wi)

        id = np.arange(m)
        cur=0
        e=[]
        for t in range(T):
            np.random.shuffle(id) 
            r_t = r if r_ is None else r_[t]
            for i in id:
                x = X[i,:].reshape((1,-1)) 
                score, ca = self.forward(x)

                dW1, db1, dW2, db2, dW3, db3 = self.backwards(x, y[i], score, ca)

                self.W1 -= r_t * dW1
                self.b1 -= r_t * db1 
                self.W2 -= r_t * dW2
                self.b2 -= r_t * db2 
                self.W3 -= r_t * dW3 
                self.b3 -= r_t * db3 

            scores, _ = self.forward(X)
            new = self.Error(scores, y)
            diff = abs(cur - new)
            cur = new
            e.append(new)

            if diff < th:
                break
        return e


    def fit(self, X):
        score, _ = self.forward(X)
        return np.sign(score.flatten())
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def Error(self, scores, y):
        return np.mean(0.5 * ((scores - y)**2))
   


def pred_cal(y):
    ycp = y.copy()
    ycp[ycp==0] = -1
    return ycp

def error(pred, y):
    return np.sum(pred!=y) / y.shape[0]



train_data = pd.read_csv('./bank-note/train.csv', header=None)
test_data = pd.read_csv('./bank-note/test.csv', header=None)
train_x = train_data.iloc[:, 0:4].to_numpy()
train_y = pred_cal(train_data.iloc[:, 4].to_numpy())
test_x = test_data.iloc[:, 0:4].to_numpy()
test_y = pred_cal(test_data.iloc[:, 4].to_numpy())



s_2ap = np.array([[-2.437]])
ca_2ap = (np.array([[-6, 6]]), np.array([[0.00247, 0.9975]]), np.array([[-4, 4]]), np.array([[0.01803, 0.9820]]))

back_p= np.array([[0.00105, 0.00158], [0.00105, 0.00158]]), np.array([0.00105, 0.00158]), np.array([[-0.0003017, 0.000226], [-0.1217, 0.0910]]), np.array([-0.122, 0.09125]), np.array([[-0.06197], [-3.375]]), np.array([-3.4369])

n_2a = NN(2)
X_2a = np.array([[1,1]]) 
y_2a = np.array([1])
m_2a, d_2a = X_2a.shape
n_2a.W1 = np.array([[-2,2],[-3,3]])
n_2a.b1 = np.array([-1, 1])
n_2a.W2 = np.array([[-2,2],[-3,3]]) 
n_2a.b2 = np.array([-1, 1]) 
n_2a.W3 = np.array([[2],[-1.5]]) 
n_2a.b3 = np.array([-1]) 

s_2a, ca_2a = n_2a.forward(X_2a)
back = n_2a.backwards(X_2a, y_2a, s_2a, ca_2a)

print('Forward pass by hand:')
print('Score',s_2ap)
print('S1:',ca_2ap[0])
print('Z1:',ca_2ap[1])
print('S2:',ca_2ap[2])
print('Z2:',ca_2ap[3])
print('Forward pass results:')
print('Score',s_2a)
print('S1:',ca_2a[0])
print('Z1:',ca_2a[1])
print('S2:',ca_2a[2])
print('Z2:',ca_2a[3])

print('Backward pass by hand:')
print('dW1:',back_p[0])
print('db1:',back_p[1])
print('dW2:',back_p[2])
print('db2:',back_p[3])
print('dW3:',back_p[4])
print('db3:',back_p[5])

print('Backward pass results:')
print('dW1:',back[0])
print('db1:',back[1])
print('dW2:',back[2])
print('db2:',back[3])
print('dW3:',back[4])
print('db3:',back[5])


r0 = 0.1
d = 0.1
T = 100
t = np.arange(T)
r_ = r0 / (1 + (r0/d)*t)

wid_l = np.array([5, 10, 25, 50, 100])

for wid in wid_l:
    net = NN(wid)
    e = net.train(train_x, train_y, wi=True, r_=r_)
    plt.figure()
    plt.plot(e, label='Training Set Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch width={0}'.format(wid))
    plt.legend()
    plt.savefig('./2a'+str(wid)+'.png')

    train_pred = net.fit(train_x)
    train_err = error(train_pred, train_y)
    y_test_pred = net.fit(test_x)
    test_err = error(y_test_pred,test_y)
    print("Width: ",wid)
    print('Train error: ',train_err)
    print('Test error: ',test_err)


r0 = 0.1
d = 0.1
T = 100
t = np.arange(T)+1
r_ = r0 / (1 + (r0/d)*t)

wid_l = np.array([5, 10, 25, 50, 100])

for wid in wid_l:
    net = NN(wid)
    e = net.train(train_x,train_y, wi=False, r_=r_)
    plt.figure()
    plt.plot(e, label='Training Set Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch width={0}'.format(wid))
    plt.legend()
    plt.savefig('./2b'+str(wid)+'.png')

    train_pred = net.fit(train_x)
    train_err = error(train_pred, train_y)
    y_test_pred = net.fit(test_x)
    test_err = error(y_test_pred,test_y)
    print("Width: ",wid)
    print('Train error: ',train_err)
    print('Test error: ',test_err)