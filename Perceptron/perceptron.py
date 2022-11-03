import pandas as pd
import numpy as np

def standard(train, test, T,lr):
    wei = np.zeros(len(train.columns))
    test_err = dict.fromkeys(i for i in range(1, T))
    test['1'] = 1
    col = test.pop('1')
    test.insert(0, '1', col)
    teX = np.array(test.drop(['label'], axis=1))
    
    for e in range(1, T + 1):
        shuffle = train.sample(frac=1).reset_index(drop=True)
        shuffle['1'] = 1
        col = shuffle.pop('1')
        shuffle.insert(0, '1', col)
        X = np.array(shuffle.drop('label', axis=1))
        y = np.array(shuffle['label'])
  
        for i in range(len(shuffle)):
           
            error = y[i] * (wei.T.dot(X[i]))
            if error <= 0:
              
                wei = wei + y[i] * X[i]*lr

        pred = []
    
        for i in range(0, len(teX)):
            pred.append(np.sign(wei.T.dot(teX[i])))
        test['pred'] = pred

        test_err[e] = len(test[test['pred'] != test['label']]) / len(test)

    err_a = sum(test_err.values()) / T
    return wei, err_a


def voted(train, test, T,lr):
    err_test = 0
    test['1'] = 1
    col = test.pop('1')
    test.insert(0, '1', col)
    teX = np.array(test.drop(['label'], axis=1))
    wei = np.zeros(len(train.columns))
    W1 = [wei]
    m1 = 0
    C = dict.fromkeys(i for i in range(0, T))
    
    for e in range(1, T + 1):
        
        shuffle= train.sample(frac=1).reset_index(drop=True)
        shuffle['1'] = 1
        col = shuffle.pop('1')
        shuffle.insert(0, '1', col)
        X = np.array(shuffle.drop('label', axis=1))
        y = np.array(shuffle['label'])

        for i in range(len(X)):
            error = y[i] * (W1[m1].T.dot(X[i]))
            if error <= 0:
                W1.append(W1[m1] + y[i] * X[i]*lr)
                m1 = m1 + 1
                C[m1] = 1
            else:
                C[m1] = C[m1] + 1
        pred_f = []
        for j in range(len(teX)):
            pred = []
            for i in range(1, len(W1)):
                pred.append(C[i] * np.sign(W1[i].T.dot(teX[j])))
            pred_f.append(np.sign(sum(pred)))
        test['pred'] = pred_f
        err_test += len(test[test['pred'] != test['label']]) / len(test)

    return W1, C, (err_test / T)


def averaged(train, test, T,lr):
    
    err_test = 0
    test['1'] = 1
    col = test.pop('1')
    test.insert(0, '1', col)
    teX = np.array(test.drop(['label'], axis=1))
    wei = np.zeros(len(train.columns))
    wef = np.zeros(len(train.columns))
    for e in range(1, T + 1):
        shuffle = train.sample(frac=1).reset_index(drop=True)
        shuffle['1'] = 1
        col = shuffle.pop('1')
        shuffle.insert(0, '1', col)
        X = np.array(shuffle.drop('label', axis=1))
        y = np.array(shuffle['label'])
        for i in range(len(X)):
            err = y[i] * (wei.T.dot(X[i]))
            if err <= 0:
                wei = wei + y[i] * X[i]*lr

            wef = wef + wei

        pred_f = []
        for j in range(len(teX)):
            pred_f.append(np.sign(wef.T.dot(teX[j])))
        test['pred'] = pred_f
        err_test += len(test[test['pred'] != test['label']]) / len(test)
    return wef, (err_test/ T)


train = pd.read_csv('./bank-note/train.csv', names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])
test = pd.read_csv('./bank-note/test.csv', names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])

train.loc[train.label == 0, "label"] = -1
test.loc[test.label == 0, "label"] = -1

print("================Standard Perceptron=================")
w,e=standard(train.copy(), test.copy(), 10,0.01)
print("Learned Weight Vector", w)
print("Test Error", e)
print("================Voted Perceptron=================")
w,c,e=voted(train.copy(), test.copy(), 10,0.01)
print("Learned Weight Vector", w)
print("Counts", c)
print("Test Error", e)
print("================Averaged Perceptron=================")
w,e=averaged(train.copy(), test.copy(), 10,0.01)
print("Learned Weight Vector", w)
print("Test Error", e)