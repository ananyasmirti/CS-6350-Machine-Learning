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
depth which stores the maximum depth the tree can grow upti
''' 
class DecisionTree():
  def __init__(self, er, depth=0):
    
    self.depth = depth
    self.er=er

#make_tree function grows the tree by first storing the root node and then spliting on each feature by calling the split_at function 
#and  then storing the them at the last returing the daughter node of root.

  def make_tree(self, data, attribute, label):
    
    A=[]
    root_d= Node()
    root_d.depth=0
    root = {'data': data,'attribute': attribute, 'label': label, 'node_d': root_d}
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

  def split_at(self, new):
    
    nlist = []
    max_g = -1
    max_f=None
    attribute = new['attribute']
    label = new['label']
    node_d = new['node_d']
    data = new['data']

    s= len(data.index)

    c= Counter(data[list(label.items())[0][0]])
    
    if s > 0:
      dom_label= c.most_common(1)[0][0]

#choosing the error function
    erf=None
    if self.er =="entropy":
      erf = self.entropy
    elif self.er == "me":
      erf = self.majority_error
    elif self.er == "gini":
      erf = self.gini_index

#check for leaf node 
    e = erf(data,label)
    if e == 0 or node_d.depth == self.depth or len(attribute.items()) == 0:
        node_d.leaf=True
        if s > 0:
          node_d.label= dom_label
        return nlist
  
#calculating information gain
    for k,v in attribute.items():
        gain = 0
        for i in v:
            small = data[data[k] == i]
            p = len(small.index) / s
            gain += p * erf(small,label)
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
    newa1=copy.deepcopy(newa)
    for a in attribute[max_f]:
      childN = Node()
      childN.depth= node_d.depth+ 1
      childN.label=dom_label
      child[a] = childN
      parent = {'data': data[data[max_f] == a],'attribute': newa1, 'label': label, 'node_d': childN}
      nlist.append(parent)
    node_d.child=child
    return nlist

#Implemented the entropy function
  def entropy(self, data, label):
    l, v = list(label.items())[0]
    s = len(data.index)
    if s == 0: return 0
    ent = 0
    for i in v:
        prob = len(data[data[l] == i]) / s
        if prob != 0:
            ent -= prob * np.log2(prob)
    return ent

#Implemented the majority error
  def majority_error(self,data, label):
    c= Counter(data[list(label.items())[0][0]])
    s = len(data.index)
    if s == 0:
      return 0
    
    max1 = c.most_common(1)[0][1]
    return 1 - (max1/s)
    
#Implemented the gini index
  def gini_index(self, data,label):  
    l, v = list(label.items())[0]
    s = len(data.index)
    if s == 0:
        return 0
    gi = 0
    for i in v:
        prob = len(data[data[l] == i]) / s
        gi += prob**2
    return 1 - gi

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

columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

train_data =  pd.read_csv('./bank/train.csv', names=columns)
test_data =  pd.read_csv('./bank/test.csv', names=columns)

#stores the train labels(expected train labels)
Y_train = train_data.iloc[:, -1].values.reshape(-1,1)
#stores the test labels(expected test labels)
Y_test = test_data.iloc[:, -1].values.reshape(-1,1)

attri_num = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

for i in attri_num:
    m = train_data[i].median()
    train_data[i] = train_data[i].apply(lambda x: 1 if x > m else 0)

for i in attri_num:
    m = test_data[i].median()
    test_data[i] = test_data[i].apply(lambda x: 1 if x > m else 0)

#attributes after converting them into binary
attribute = {'age': [0, 1], 
        'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'], 
        'marital': ['married','divorced','single'], 
        'education': ['unknown', 'secondary', 'primary', 'tertiary'],
        'default': ['yes', 'no'],
        'balance': [0, 1], 
        'housing': ['yes', 'no'],
        'loan': ['yes', 'no'],
        'contact': ['unknown', 'telephone', 'cellular'],
        'day': [0, 1],  
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'duration': [0, 1],  
        'campaign': [0, 1], 
        'pdays': [0, 1], 
        'previous': [0, 1],  
        'poutcome': ['unknown', 'other', 'failure', 'success']}
label = {'y': ['yes', 'no']}

T = 500
n = 100

train_size = len(train_data.index)
test_size = len(test_data.index)

test_pred = np.array([[0 for x in range(test_size)] for y in range(n)])
t_f_pred = np.array([0 for x in range(test_size)])

for i in range(n):
    train_subset = train_data.sample(n=1000, replace=False, random_state=i)
    print('i: ', i)
    for j in range(T):
      sampled = train_subset.sample(frac=0.01, replace=True, random_state=i)
      classifier = DecisionTree(er="entropy", depth=17)
      decision_tree = classifier.make_tree(sampled, attribute, label)
      pred = classifier.make_pred(decision_tree, test_data) 
      pred = np.array(pred.tolist())
      pred[pred == 'yes'] = 1
      pred[pred == 'no'] = -1
      pred = pred.astype(int)
      test_pred[i] = test_pred[i] + pred
      if j == 0:
          t_f_pred = t_f_pred + pred

value = np.array(test_data['y'].tolist())
value[value == 'yes'] = 1
value[value == 'no'] = -1
value = value.astype(int)

t_f_pred = t_f_pred/ n
bias = np.mean(np.square(t_f_pred - value))
mean = np.mean(t_f_pred) 
variance = np.sum(np.square(t_f_pred - mean)) / (test_size - 1)
s = bias + variance
print("bias=",bias)
print("variance=",variance)
print('Sum of bias and variance in 100 single tree predictor= ', s)

test_pred = np.sum(test_pred,axis=0) / (n * T)
bias = np.mean(np.square(test_pred - value))
mean = np.mean(test_pred)
variance = np.sum(np.square(test_pred - mean)) / (test_size - 1)
s = bias + variance
print("bias=",bias)
print("variance=",variance)
print('Sum of bias and variance in 100 bagged tree predictor=', s)