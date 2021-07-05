# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import coo_matrix
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

##################################
# Work with Two Adjacency Matrix #
##################################


#construct A_train and A_test for userdip
#convert edge information into list
edge_array= np.array(data_userdip)
edge_list=list(edge_array)
A_train = Counter()
A_test = Counter()
#take t=56 as threshold
for link in edge_list:
    if link[2] <= 56:
        A_train[link[0],link[1]] = 1
    else:
        A_test[link[0],link[1]] = 1
#construct adjacency matrix in the form of sparse matrix
def counter2A(A,m=None,n=None):
    #A is a counter
    A_keys = np.array(list(A.keys()))
    if m is None:
        m = np.max(A_keys[:,0])+1
    if n is None:
        n = np.max(A_keys[:,1])+1
    A_mat = coo_matrix((np.ones(A_keys.shape[0]), (A_keys[:,0], A_keys[:,1])), shape=(m, n))
    return A_mat
    
A_train_keys = np.array(list(A_train.keys()))
A_test_keys = np.array(list(A_test.keys()))
m = np.max([np.max(A_train_keys[:,0]),np.max(A_test_keys[:,0])])+1
n = np.max([np.max(A_train_keys[:,1]),np.max(A_test_keys[:,1])])+1
A_train_mat = counter2A(A_train,m,n)
A_test_mat = counter2A(A_test,m,n)

#determine dimension of embedding
U_train,eigval_train,V_train = svds(A_train_mat,k=500)
iterate_zhu(eigval_train[::-1],x=5)
plt.figure(figsize=(7,5))
plot_d = 500
plt.scatter(np.linspace(1,plot_d,plot_d),eigval_train[::-1][:plot_d],s=2)
plt.vlines(x=25,ymin=0,ymax=200,linestyle='dashed',color='r')
plt.vlines(x=83,ymin=0,ymax=200,linestyle='dashed',color='r')
plt.vlines(x=194,ymin=0,ymax=200,linestyle='dashed',color='r')
plt.vlines(x=310,ymin=0,ymax=200,linestyle='dashed',color='r')
plt.show()

#use d=25 for training set
U_train,eigval_train,V_train = svds(A_train_mat,k=25)
#scale with square root of eigvenvalues
U_train = U_train @ np.diag(np.sqrt(eigval_train))
V_train = np.diag(np.sqrt(eigval_train)) @ V_train

#use d=25 for test set
U_test,eigval_test,V_test = svds(A_test_mat,k=25)
#scale with square root of eigvenvalues
U_test = U_test @ np.diag(np.sqrt(eigval_test))
V_test = np.diag(np.sqrt(eigval_test)) @ V_test

#resampling to have same number of 0 and 1
def resampling(A,U,V,size_multiplier = 2):
    ## Sample of the negative class
    A_keys = np.array(list(A.keys()))
    m = np.max(A_keys[:,0])+1
    n = np.max(A_keys[:,1])+1
    n_edges = len(A)
    negative_class = np.zeros((int(size_multiplier * n_edges),2))
    negative_class[:,0] = np.random.choice(m, size=negative_class.shape[0])
    negative_class[:,1] = np.random.choice(n, size=negative_class.shape[0])
    ## delete repeated pairs
    negative_class = np.unique(negative_class,axis=0)
    ## convert into counter
    negative_class_counter = Counter()
    for pair in list(negative_class):
        negative_class_counter[pair[0],pair[1]] = 0
    ## Check that the sampled elements effectively correspond to the negative class
    negative_class_indices = np.array([pair not in A for pair in negative_class_counter])
    negative_class = negative_class[negative_class_indices]
    return negative_class

#use hadamard to construct edge feature
def ef_hadamard(A,U,V,negative_class):
    n_edges = len(A)
    ## Calculate the scores for negative_class
    scores_negative_class = []
    for pair in negative_class[:n_edges,:]:
        scores_negative_class += [U[int(pair[0]),:]*V[:,int(pair[1])] ]
    ## Calculate the scores for positive_class
    scores_positive_class = []
    for pair in A:
        scores_positive_class += [U[int(pair[0]),:]*V[:,int(pair[1])] ]
    #combine for x and y
    x = np.vstack((np.array(scores_negative_class),np.array(scores_positive_class)))
    y = np.hstack((np.zeros(len(scores_negative_class)),np.ones(len(scores_positive_class))))
    return x,y
    
#use average to construct edge feature
def ef_average(A,U,V,negative_class):
    n_edges = len(A)
    ## Calculate the scores for negative_class
    scores_negative_class = []
    for pair in negative_class[:n_edges,:]:
        scores_negative_class += [(U[int(pair[0]),:]+V[:,int(pair[1])])*0.5]
    ## Calculate the scores for positive_class
    scores_positive_class = []
    for pair in A:
        scores_positive_class += [(U[int(pair[0]),:]+V[:,int(pair[1])])*0.5]
    #combine for x and y
    x = np.vstack((np.array(scores_negative_class),np.array(scores_positive_class)))
    y = np.hstack((np.zeros(len(scores_negative_class)),np.ones(len(scores_positive_class))))
    return x,y

#using Hadamard
#construct training set based on A_train
train_negative_class = resampling(A_train,U_train,V_train)
x_train,y_train = ef_hadamard(A_train,U_train,V_train,train_negative_class)
#construct test set based on A_test
test_negative_class = resampling(A_test,U_test,V_test)
x_test,y_test = ef_hadamard(A_test,U_test,V_test,test_negative_class)
rf_classfier(x_train,y_train,x_test,y_test)
lr_classfier(x_train,y_train,x_test,y_test)


#using average
train_negative_class = resampling(A_train,U_train,V_train)
x_train,y_train = ef_average(A_train,U_train,V_train,train_negative_class)
test_negative_class = resampling(A_test,U_test,V_test)
x_test,y_test = ef_average(A_test,U_test,V_test,test_negative_class)
print(rf_classfier(x_train,y_train,x_test,y_test))
print(lr_classfier(x_train,y_train,x_test,y_test))


#random dot product graph RDPG
def rdpg(A,U,V,negative_class):
    n_edges = len(A)
    
    ## Calculate the scores for negative_class
    scores_negative_class = []
    for pair in negative_class[:n_edges,:]:
        scores_negative_class += [np.dot(U[int(pair[0]),:],V[:,int(pair[1])]) ]
        
    ## Calculate the scores for positive_class
    scores_positive_class = []
    for pair in A:
        scores_positive_class += [np.dot(U[int(pair[0]),:],V[:,int(pair[1])]) ]
    #combine for x and y
    x = np.concatenate((np.array(scores_negative_class),np.array(scores_positive_class)))
    y = np.concatenate((np.zeros(len(scores_negative_class)),np.ones(len(scores_positive_class))))
    return x,y

auc_pred= []
l=100
for i in range(l):
    negative_class_A_test = resampling(A_test)
    x,y=rdpg(A_test,U_train,V_train,negative_class_A_test)
    auc_pred.append(sklearn.metrics.roc_auc_score(y,x))

plt.plot(np.linspace(1,100,100),auc_pred,'o-')
plt.xlabel('Times')
plt.ylabel('AUC')
plt.show()
