# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score
#import functions from files
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
plt.title('Plot of eigenvalues by SVD')
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

#repeating the resampling for size_multiplier = 1,.....20
for j in range(10):
   auc_pred2= []
   l=20
   for i in range(1,l+1):
       negative_class_A_test = lanl.resampling(A_test,size_multiplier = i)
       x,y=lanl.rdpg(A_test,U_train,V_train,negative_class_A_test)
       auc_pred2.append(roc_auc_score(y,x))
   plt.plot(np.linspace(1,l,l),auc_pred2,'o-')
plt.xlabel('Size Multiplier')
plt.ylabel('AUC score')
plt.title('The AUC score using RGPD via different size_multiplier')
plt.show()


########################################
# Work with Multiple Adjacency Matrices#
########################################

#construct adjacency matrix for time 1 to 90 days
A = lanl.counter_A(data_userdip,multiple = True,direct=True)
m,n = lanl.find_dimension(data_userdip) #find dimension of adjacency matrix
A_mat = {}

for i in range(len(A)):
    A_mat[i] = lanl.counter2A(A[i],m,n)

#determine dimension d of embedding: use d from average of A_t
#choose t=56 as threshold
for i in range(56):
    if i==0:
        A_average = A_mat[0]
    else:
        A_average += A_mat[i] 

A_average = A_average/56
U_av,eigval_av,V_av_T = svds(A_average,k=500)
zhu_d=zhu.iterate_zhu(eigval_av[::-1],x=5)#use d=22
plt.figure(figsize=(7,5))
plot_d = 500
plt.scatter(np.linspace(1,plot_d,plot_d),eigval_av[::-1],s=2)
plt.vlines(x=zhu_d[1],ymin=0,ymax=np.max(eigval_av),linestyle='dashed',color='r')
plt.vlines(x=zhu_d[2],ymin=0,ymax=np.max(eigval_av),linestyle='dashed',color='r')
plt.vlines(x=zhu_d[3],ymin=0,ymax=np.max(eigval_av),linestyle='dashed',color='r')
plt.vlines(x=zhu_d[4],ymin=0,ymax=np.max(eigval_av),linestyle='dashed',color='r')
plt.xlabel('dimensions')
plt.ylabel('eigvenvalues')
plt.title('Plot of eigenvalues by SVD')
plt.show()



#########
# COSIE #
#########

#compute R_t using MASE for directed graph
hat_X,hat_Y,R = lanl.mase_direct(A_mat,22)

#compute R_average
R_average = lanl.average_mat(R,length=56)

#resampling for t=57
negative_class = lanl.resampling(A[56],size_multiplier = 2)
x_cosie,y_cosie = lanl.cosie_average(A[56],hat_X,hat_Y,R_average,negative_class)

#compute auc for prediction of A at t=57, repeat 100 times
auc_pred3= []
l=100
for i in range(l):
    negative_class = lanl.resampling(A[56],size_multiplier = 2)
    x_cosie,y_cosie = lanl.cosie_average(A[56],hat_X,hat_Y,R_average,negative_class)
    auc_pred3.append(roc_auc_score(y_cosie,x_cosie))

plt.plot(np.linspace(1,l,l),auc_pred3,'o-')
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using COSIE')
plt.show()


#repeating the resampling for size_multiplier = 1,.....30
for j in range(10):
   auc_pred4= []
   l=30
   for i in range(1,l+1):
    negative_class = lanl.resampling(A[56],size_multiplier = 2)
    x_cosie,y_cosie = lanl.cosie_average(A[56],hat_X,hat_Y,R_average,negative_class)
    auc_pred4.append(roc_auc_score(y_cosie,x_cosie))
   plt.plot(np.linspace(1,l,l),auc_pred4,'o-')
plt.xlabel('Size Multiplier')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using COSIE via different size_multiplier')
plt.show()



    
    
    
    






