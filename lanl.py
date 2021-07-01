# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from scipy.stats import norm
from scipy.sparse.linalg import svds,eigsh
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# load data as pandas data frames
data_userdip= pd.read_csv('userdip.txt',header=None)
data_usersip= pd.read_csv('usersip.txt',header=None)



#construct directed sparse adjacency matrix
from collections import Counter
from scipy.sparse import coo_matrix
def direct_sparse_A(data):
    #convert edge information into list
    edge_array= np.array(data)
    edge_list=list(edge_array)
    A = Counter()
    #directed graph (symmetric A)
    for link in edge_list:
        A[link[0],link[1]] = 1
    #construct adjacency matrix in the form of sparse matrix
    A_keys = np.array(list(A.keys()))
    m = np.max(A_keys[:,0])+1
    n = np.max(A_keys[:,1])+1
    A_mat = coo_matrix((np.ones(A_keys.shape[0]), (A_keys[:,0], A_keys[:,1])), shape=(m, n))
    return A_mat

A_userdip = direct_sparse_A(data_userdip)
A_usersip = direct_sparse_A(data_usersip)




#functions by zhu to find d
def zhu(d):
    p = len(d)
    profile_likelihood = np.zeros(p)
    for q in range(1,p-1):
        mu1 = np.mean(d[:q])
        mu2 = np.mean(d[q:])
        sd = np.sqrt(((q-1) * (np.std(d[:q]) ** 2) + (p-q-1) * (np.std(d[q:]) ** 2)) / (p-2))
        profile_likelihood[q] = norm.logpdf(d[:q],loc=mu1,scale=sd).sum() + norm.logpdf(d[q:],loc=mu2,scale=sd).sum()
    return profile_likelihood[1:p-1], np.argmax(profile_likelihood[1:p-1])+1

def iterate_zhu(d,x=3):
    results = np.zeros(x,dtype=int)
    results[0] = zhu(d)[1]
    for i in range(x-1):
        results[i+1] = results[i] + zhu(d[results[i]:])[1]
    return results

U1,eigval1,V1 = svds(A_userdip,k=5000)
plt.figure(figsize=(7,5))
plot_d = 5000
plt.scatter(np.linspace(1,plot_d,plot_d),eigval1[::-1][:plot_d],s=2)
plt.vlines(x=39,ymin=0,ymax=200,linestyle='dashed',color='r')
plt.show()
plt.figure(figsize=(7,5))
plot_d = 100
plt.scatter(np.linspace(1,plot_d,plot_d),eigval1[::-1][:plot_d],s=2)
plt.vlines(x=39,ymin=0,ymax=200,linestyle='dashed',color='r')
plt.show()

#choose d=39
U_userdip,eigval_userdip,V_userdip = svds(A_userdip,k=39)


#construct edge feature: using Hadarmard
edge_feature_hadarmard = np.zeros(A_userdip.shape)
for i in range(12222):
    u_i = U_userdip[i,:]
    for j in range(5047):
        v_j = V_userdip[:,j]
        edge_feature_hadarmard[i,j] = np.dot(u_i,v_j) 

#flattent the matrix
flatten_edge_feature_hadarmard=edge_feature_hadarmard.reshape(edge_feature_hadarmard.size,1)
flatten_A_userdip = A_userdip.todense()
flatten_A_userdip = np.array(flatten_A_userdip).reshape(edge_feature_hadarmard.size,1)
#combine explanory matrix and its label
userdip_set = np.hstack((flatten_edge_feature_hadarmard,flatten_A_userdip))


#find index of 1 and 0 separately (time-consuming algorithm)
index1 = np.array([])
index0 = np.array([])

for i in range(len(userdip_set)):
    if flatten_A_userdip[i]==1:
        index1 = np.append(index1,i)
    else:
        index0 = np.append(index0,i)


#using dictionary (incomplete)
#dictionary of pair and label 1/0
dict1 = {str(i):flatten_A_userdip[i] for i in range(len(flatten_A_userdip)) }
#dictionary of pair and edge feature
dict2 = {str(i):flatten_edge_feature_hadarmard[i] for i in range(len(flatten_edge_feature_hadarmard)) }

def get_dict_key(dic, value):
    keys = list(dic.keys())
    values = list(dic.values())
    idx = values.index(value)
    key = keys[idx]
    return key


#construct training set and test set:
def construct_train_test(data,index1,index0,split_ratio=0.7):
    
    #ensure there are same number of label1 and label0
    chose_index0 = index0[random.sample(range(len(index0)),len(index1))]
    
    #combine two types of label index
    index_set = np.hstack((chose_index0,index1))
    random.shuffle(index_set) #shuffle the data
    
    #chosen dataset:
    chosen_set = np.zeros((len(index_set),2))
    for i in range(len(index_set)):
        idx = index_set[i]
        chosen_set[i,:] = data[int(idx),:]
    
    #split into training set and test set
    split_index = int(len(index_set)*split_ratio)
    training_set = chosen_set[:split_index,:]
    test_set = chosen_set[split_index:,:]
    
    return training_set[:,0].reshape(-1,1),training_set[:,1],test_set[:,0].reshape(-1,1),test_set[:,1]

x_train,y_train,x_test,y_test = construct_train_test(userdip_set,index1,index0)

#using random forest:
def compute_auc(y, pred_y):
    fpr,tpr,threshold = roc_curve(y, pred_y)
    roc_auc = auc(fpr,tpr)
    return roc_auc

def rf_classfier(x_train,y_train,x_test,y_test):
    rfc = RandomForestClassifier()
    rfc.fit(x_train,y_train)
    #prediction of training set
    pred_y_train=rfc.predict(x_train)
    roc_auc1 = compute_auc(y_train, pred_y_train)
    #prediction of test set
    pred_y_test=rfc.predict(x_test)
    roc_auc2 = compute_auc(y_test, pred_y_test)
  
    return roc_auc1,roc_auc2

rf_classfier(x_train,y_train,x_test,y_test)

#using logit regression 
def lr_classfier(x_train,y_train,x_test,y_test):
    lr = LogisticRegression(random_state=0).fit(x_train,y_train)
    
    #prediction of training set
    pred_y_train=lr.predict(x_train)
    roc_auc1 = compute_auc(y_train, pred_y_train)
    #prediction of test set
    pred_y_test=lr.predict(x_test)
    roc_auc2 = compute_auc(y_test, pred_y_test)
  
    return roc_auc1,roc_auc2

lr_classfier(x_train,y_train,x_test,y_test)

