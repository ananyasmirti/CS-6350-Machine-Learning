import numpy as np
import pandas as pd
from collections import Counter
import copy

'''
Created a class Node which will store information of my tree Nodes.
  attribute: Has the features of the node
  child: Has the daughter node
  leaf: Tell whether this node is a leaf or not
  depth: Stores the depth at which node is
  label: Stores the label of the node
'''
class Node():
  def __init__(self, attribute=None, child=None, leaf=False, depth=None, label=None):
    self.attribute=attribute
    self.child= child
    self.leaf = leaf
    self.depth=depth
    self.label=label

'''
Here we implement the ID3 alogorithm class DecisionTree which takes the error funtion in er by which it will compute einformation gain, 
depth which stores the maximum depth the tree can grow upto
''' 
class DecisionTree():
  def __init__(self, er, depth=0):
    
    self.depth = depth
    self.er=er

#make_tree function grows the tree by first storing the root node and then spliting on each feature by calling the split_at function 
#and  then storing the them at the last returing the daughter node of root.
#Weights are taken in consideration here. 

  def make_tree(self, data, attribute, label,weights):
    
    A=[]
    root_d= Node()
    root_d.depth=0
    root = {'data': data,'weights': weights,'attribute': attribute, 'label': label, 'node_d': root_d}
    A.append(root)
    while len(A)>0:
      new=A.pop(0)
      node= self.split_at(new)
      for n in node:
        A.append(n)
    return root_d

#This fuction splits on the attributes based on the best maximum information gain by choosing the error fuction given by the user at input and returns 
#the node list for the decision tree to built on
#For leaf node it assigns them the value by assigning to the most common label
#--change-- using weights to get the most important label

  def split_at(self, new):
    
    nlist = []
    max_g = -1
    max_f=None
    attribute = new['attribute']
    label = new['label']
    node_d = new['node_d']
    data = new['data']
    weights = new['weights']

    s= sum(weights)
    dom_label = self.max_lab(data, label, weights)

#choosing the error function
    erf=None
    if self.er =="entropy":
      erf = self.entropy
    elif self.er == "me":
      erf = self.majority_error
    elif self.er == "gini":
      erf = self.gini_index

#check for leaf node 
    e = erf(data,label,weights)
    if e == 0 or node_d.depth == self.depth or len(attribute.items()) == 0:
        node_d.leaf=True
        if s > 0:
          node_d.label= dom_label
        return nlist
  
#calculating information gain now considering the weights
    for k,v in attribute.items():
        column = np.array(data[k].tolist())
        gain = 0
        for i in v:
            w = weights[column == i]
            sub_weights = w
            p = np.sum(sub_weights) / s
            subset = data[data[k] == i]
            gain += p * erf(subset, label, sub_weights)
      
        gain = e - gain
        if gain > max_g:
            max_g = gain
            max_f = k
 
    child={}
    node_d.attribute= max_f

#After we get the max info gain feature it is the parent node then assigning the daughter node it's most common label and storing the parent in a list
#Returning the list of parent nodes

    newa= copy.deepcopy(attribute)
    newa.pop(max_f, None)
    column = np.array(data[max_f].tolist())
    newa1=copy.deepcopy(newa)
    for a in attribute[max_f]:
      childN = Node()
      childN.depth= node_d.depth+ 1
      childN.label=dom_label
      child[a] = childN
      w = weights[column == a]
      parent = {'data': data[data[max_f] == a],'weights': w,'attribute': newa1, 'label': label, 'node_d': childN}
      nlist.append(parent)
    node_d.child=child
    return nlist

#Implemented the entropy function
  def entropy(self, data, label, weights):
    l, v = list(label.items())[0]
    s = np.sum(weights)
    column = np.array(data[l].tolist())
    if s == 0: return 0
    ent = 0
    for i in v:
        w = weights[column == i]
        prob = np.sum(w) / s
        if prob != 0:
            ent -= prob * np.log2(prob)
    return ent

#Implemented the majority error
  def majority_error(self,data, label,weights):
    l, v = list(label.items())[0]
    s = np.sum(weights)
    column = np.array(data[l].tolist())
    if s == 0:
        return 0
    max_p = 0
    for i in v:
        w = weights[column == i]
        p = np.sum(w) / s
        max_p = max(max_p, p)
    return 1 - max_p
    
#Implemented the gini index
  def gini_index(self, data,label,weights):  
    l, v = list(label.items())[0]
    s = np.sum(weights)
    column = np.array(data[l].tolist())
    if s == 0:
        return 0
    gi = 0
    for i in v:
        w = weights[column == i]
        prob = np.sum(w) / s
        
        gi += prob**2
    return 1 - gi

#Here we are implementing the function which can select the maximum weight label
  def max_lab(self, data, label, weights):
    l, v = list(label.items())[0]
    max_l = None
    msum = -1
    column = np.array(data[l].tolist())
    for i in v:
        w = weights[column == i]
        wsum = np.sum(w)
        if wsum > msum:
            max_l = i
            msum = wsum
    
    return max_l

#predicts the labels on the tree by checking if it encounter a leaf node it sees the label in it's daughter node
  def predict(self, tree, test):
    t = tree
    while t.leaf!=True: 
        t = t.child[test[t.attribute]]
    return t.label

#applies the predict fucntion of the entire test data
  def make_pred(self, tree, test):
    pred= test.apply(lambda row: self.predict(tree, row), axis=1)
    return pred

#calculates the accuracy error by comparing the excepted labels with the predicted labels  
  def test_error(self, pred_labels, ex_labels):
    count = 0
    for pl, el in zip(pred_labels, ex_labels):
      if pl == el:
        count += 1
    return 1 - count/len(ex_labels)

import matplotlib.pyplot as plt
import math

columns = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','y']

train_data =  pd.read_csv('./credit/train.csv', names=columns)
test_data =  pd.read_csv('./credit/test.csv', names=columns)

#stores the train labels(expected train labels)
Y_train = train_data.iloc[:, -1].values.reshape(-1,1)
#stores the test labels(expected test labels)
Y_test = test_data.iloc[:, -1].values.reshape(-1,1)

attri_num = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']

for i in attri_num:
    m = train_data[i].median()
    train_data[i] = train_data[i].apply(lambda x: 1 if x > m else 0)

for i in attri_num:
    m = test_data[i].median()
    test_data[i] = test_data[i].apply(lambda x: 1 if x > m else 0)

#attributes after converting them into binary
attribute = {'X1': [0, 1],  
        'X2': [1, 2], 
        'X3': [0,1,2,3,4,5,6], 
        'X4': [0,1,2,3],
        'X5': [0, 1],
        'X6': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'X7': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'X8': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'X9': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'X10': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'X11': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'X12': [0, 1],
        'X13': [0, 1],
        'X14': [0, 1],
        'X15': [0, 1],
        'X16': [0, 1],
        'X17': [0, 1],
        'X18': [0, 1],
        'X19': [0, 1],
        'X20': [0, 1],
        'X21': [0, 1],
        'X22': [0, 1],
        'X23': [0, 1],}
label = {'y': [0, 1]}


T = 500

train_size = len(train_data.index)
test_size = len(test_data.index)



#initializing the variables to be used 
adat = [0 for x in range(T)]
weights = np.array([1 / train_size for x in range(train_size)])

e_train = [0 for x in range(T)]
e_test = [0 for x in range(T)]
c_e_train = [0 for x in range(T)]
c_e_test  = [0 for x in range(T)]

train_pred = np.array([0 for x in range(train_size)])
test_pred = np.array([0 for x in range(test_size)])

for i in range(T):

    classifier = DecisionTree(er="entropy",depth=1)
    decision_tree = classifier.make_tree(train_data, attribute, label, weights)

    #calciulating the train error on individual tree
    train_data['pred'] = classifier.make_pred(decision_tree, train_data)
    g = train_data.apply(lambda row: 1 if row['y'] == row['pred'] else 0, axis=1) 
    error = 1 - g.sum() / train_size
    e_train[i] = error
    
    #Here we are calculating the weighted error and the alpha(ada)
    g = train_data.apply(lambda row: 1 if row['y'] == row['pred'] else -1, axis=1) 
    g = np.array(g.tolist())
    w = weights[g == -1]
    error = np.sum(w)
    ada = 0.5 * math.log((1 - error) / error)
    adat[i] = ada

    #Initializing the new weights
    weights = np.exp(g * -ada) * weights
    s = np.sum(weights)
    weights = weights / s

    #calciulating the test error on individual tree
    test_data['pred']= classifier.make_pred(decision_tree, test_data)
    g = test_data.apply(lambda row: 1 if row['y'] == row['pred'] else 0, axis=1) 
    e_test[i] = 1 - g.sum() / test_size

    #calciulating the train error on combined
    pred = np.array(train_data['pred'].tolist())
    pred[pred == 1] = 1
    pred[pred == 0] = -1
   
    train_pred = train_pred + pred * ada
   
    pred[train_pred > 0] = 1
    pred[train_pred <=0] = 0
    train_data['pred'] = pd.Series(pred)
    g = train_data.apply(lambda row: 1 if row['y'] == row['pred'] else 0, axis=1) 
    error = 1 - g.sum() / train_size
    c_e_train[i] = error

    #calciulating the test error on combined
    pred = np.array(test_data['pred'].tolist())
    pred[pred == 1] = 1
    pred[pred == 0] = -1
    
    test_pred = test_pred + pred * ada
    
    pred[test_pred > 0] = 1
    pred[test_pred <=0] = 0
    test_data['pred'] = pd.Series(pred)
    g1 = test_data.apply(lambda row: 1 if row['y'] == row['pred'] else 0, axis=1) 
    error1 = 1 - g1.sum() / test_size
    c_e_test[i] = error1

    print('iter: ', i, 'error_I_train: ', e_train[i], 'error_I_test: ', e_test[i], 'error_C_train', c_e_train[i], 
          'error_C_test:', c_e_test[i]) 

func1 , (axis1, axis2) = plt.subplots(2,constrained_layout = True)
axis1.plot(e_train, 'b')
axis1.plot(e_test, 'r')  
axis1.legend(['train', 'test'])
axis1.set_title('Individual Tree Predictor')
axis1.set_xlabel('No.of iteration', fontsize=12)
axis1.set_ylabel('Error', fontsize=12)
axis2.plot(c_e_train, 'b')
axis2.plot(c_e_test, 'r')  
axis2.legend(['train', 'test'])
axis2.set_title('Total Predictor')
axis2.set_xlabel('No.of iteration', fontsize=12)
axis2.set_ylabel('Error', fontsize=12)
func1.savefig('adaC.png')



