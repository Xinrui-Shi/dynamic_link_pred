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
edge_list=list(edge_array) #### %% FSP: this is not needed, you could simply do the loop 'for link in edge_array:'
A_train = Counter()
A_test = Counter()
#take t=56 as threshold
for link in edge_list:
    if link[2] <= 56:
        A_train[link[0],link[1]] = 1
    else:
        A_test[link[0],link[1]] = 1

#### %% FSP: Why is this repeated twice?
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
zhu.iterate_zhu(eigval_train[::-1],x=5) #### %% FSP: note that the Zhu & Ghodsi criterion also depends on your choice of k above!
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
plt.show() #### %% FSP: Nice, I like this plot!

#use d=25 for training set
U_train,eigval_train,V_train = svds(A_train_mat,k=25)
#scale with square root of eigvenvalues
U_train = U_train @ np.diag(np.sqrt(eigval_train))
V_train = np.diag(np.sqrt(eigval_train)) @ V_train


#repeating the resampling for 100 times, usingsize_multiplier = 2
auc_pred= []
l=100
for i in range(l):
    print('\rIteration: ', str(i+1), ' / ', str(l), sep='', end='') #### %% FSP: suggestion to monitor the progress
    negative_class_A_test = lanl.resampling(A_test,size_multiplier = 2)
    x,y=lanl.rdpg(A_test,U_train,V_train,negative_class_A_test)
    auc_pred.append(roc_auc_score(y,x))

plt.plot(np.linspace(1,l,l),auc_pred,'o-')
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using RGPD (using A_train)')
plt.show()

#### %% FSP: The simulation for different values of the size multiplier should be aimed at understanding how the variance of the AUC scores changes
#### %% FSP: At the moment the simulation does not show that. I tried to modify the code a little bit

#repeating the resampling for size_multiplier = 1,.....100
auc_pred2 = {} ## dictionary
l = [0.1,0.2,0.5,1,2] ## list of resampling sizes
n_iter = 10 ## number of iterations per resampling size
## For each resampling size
for i in l:
    ## The 'value' of the dictionary for the 
    auc_pred2[i] = []
    ## Repeat for n_iter iterations
    for _ in range(n_iter):
        print('\rSize multiplier: ', str(i), '\tIteration: ', str(_+1), ' / ', str(n_iter), sep='', end='')
        negative_class_A_test = lanl.resampling(A_test,size_multiplier = i)
        x,y = lanl.rdpg(A_test,U_train,V_train,negative_class_A_test)
        auc_pred2[i].append(roc_auc_score(y,x))

## Plot each AUC as a line to show that the variance of the points decreases (or calculate the variance directly and plot the estimates)
for j in range(len(l)):
    plt.plot(j * np.ones(n_iter),auc_pred2[l[j]],'o-')

plt.xlabel('Size Multiplier')
plt.ylabel('AUC score')
plt.title('The AUC score using RGPD via different size_multiplier')
plt.show()

#### %% FSP: **** VERY IMPORTANT **** The results show that (most likely) there are some issues with the function rdpg. See comments for that function.

########################################
# Work with Multiple Adjacency Matrices#
########################################

#construct adjacency matrix for time 1 to 90 days
A = lanl.counter_A(data_userdip,multiple = True,direct=True)
m,n = lanl.find_dimension(data_userdip) #find dimension of adjacency matrix
A_mat = {}

for i in range(len(A)):
    A_mat[i] = lanl.counter2A(A[i],m,n)

#determine dimension d of embedding 
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
R = lanl.mase_direct(A_mat,22)
















