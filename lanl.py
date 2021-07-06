# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score
#import functions from filessx
import lanl_function as lanl
import zhu

# load data as pandas data frames
data_userdip= pd.read_csv('userdip.txt',header=None)
data_usersip= pd.read_csv('usersip.txt',header=None)


##################################
# Work with Two Adjacency Matrix #
##################################


#construct A_train and A_test for userdip
#convert edge information into list
from collections import Counter
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

#take t=56 as threshold
for link in edge_list:
    if link[2] <= 56:
        A_train[link[0],link[1]] = 1
    else:
        A_test[link[0],link[1]] = 1
        
#construct adjacency matrix in the form of sparse matrix
m,n = lanl.find_dimension(data_userdip)
A_train_mat = lanl.counter2A(A_train,m,n)
A_test_mat = lanl.counter2A(A_test,m,n)

#determine dimension of embedding
U_train,eigval_train,V_train = svds(A_train_mat,k=500)
zhu.iterate_zhu(eigval_train[::-1],x=5)
plt.figure(figsize=(7,5))
plot_d = 500
plt.scatter(np.linspace(1,plot_d,plot_d),eigval_train[::-1][:plot_d],s=2)
plt.vlines(x=25,ymin=0,ymax=200,linestyle='dashed',color='r')
plt.vlines(x=83,ymin=0,ymax=200,linestyle='dashed',color='r')
plt.vlines(x=194,ymin=0,ymax=200,linestyle='dashed',color='r')
plt.vlines(x=310,ymin=0,ymax=200,linestyle='dashed',color='r')
plt.xlabel('dimensions')
plt.ylabel('eigvenvalues')
plt.plot('Plot of eigenvalues by SVD')
plt.show()

#use d=25 for training set
U_train,eigval_train,V_train = svds(A_train_mat,k=25)
#scale with square root of eigvenvalues
U_train = U_train @ np.diag(np.sqrt(eigval_train))
V_train = np.diag(np.sqrt(eigval_train)) @ V_train


#repeating the resampling for 100 times, usingsize_multiplier = 2
auc_pred= []
l=100
for i in range(l):
    negative_class_A_test = lanl.resampling(A_test,size_multiplier = 2)
    x,y=lanl.rdpg(A_test,U_train,V_train,negative_class_A_test)
    auc_pred.append(roc_auc_score(y,x))

plt.plot(np.linspace(1,l,l),auc_pred,'o-')
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using RGPD (using A_train)')
plt.show()

#repeating the resampling for size_multiplier = 1,.....100
auc_pred2= []
l=50
for i in range(1,l+1):
    negative_class_A_test = lanl.resampling(A_test,size_multiplier = i)
    x,y=lanl.rdpg(A_test,U_train,V_train,negative_class_A_test)
    auc_pred2.append(roc_auc_score(y,x))

plt.plot(np.linspace(1,l,l),auc_pred2,'o-')
plt.xlabel('Size Multiplier')
plt.ylabel('AUC score')
plt.title('The AUC score using RGPD via different size_multiplier')
plt.show()







