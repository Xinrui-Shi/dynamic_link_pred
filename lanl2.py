#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:34:01 2021

@author: shining
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
#import functions from files
import lanl_function as lanl

# load data as pandas data frames
data_userdip= pd.read_csv('userdip.txt',header=None)

#construct adjacency matrix for time 1 to 90 days
A = lanl.counter_A(data_userdip,multiple = True,direct=True)
m,n = lanl.find_dimension(data_userdip) #find dimension of adjacency matrix
A_mat = {}

for i in range(len(A)):
    A_mat[i] = lanl.counter2A(A[i],m,n)


A_observed = {}
for t in range(56):
    A_observed[t]=A[t]


#AIP 
X = {}
Y = {}
for t in range(56):
    X_t,Y_t = lanl.dase(A_mat[t],d=22)
    X[t] = X_t
    Y[t] = Y_t

#compute auc for prediction of A at t=57, repeat 100 times
auc_pred5= []
l=20
for i in range(l):
    print('\rIteration: ', str(i+1), ' / ', str(l), sep='', end='')
    positive_class,negative_class = lanl.resampling2(A_observed,A[56],size_multiplier = 1)
    x,y = lanl.aip(positive_class,X,Y,negative_class)
    auc_pred5.append(roc_auc_score(y,x))

plt.plot(np.linspace(1,l,l),auc_pred5,'o-')
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using AIP on Multiple Adjacency Matrice')
plt.show()

#repeating the resampling for size_multiplier = 0.1,0.2,0.5,1,2
auc_pred6 = {} ## dictionary
l = [0.1,0.2,0.5,1,2] ## list of resampling sizes
n_iter = 20 ## number of iterations per resampling size
## For each resampling size
for i in l:
    ## The 'value' of the dictionary for the 
    auc_pred6[i] = []
    ## Repeat for n_iter iterations
    for _ in range(n_iter):
        print('\rSize multiplier: ', str(i), '\tIteration: ', str(_+1), ' / ', str(n_iter), sep='', end='')
        positive_class,negative_class = lanl.resampling2(A_observed,A[56],size_multiplier = i)
        x,y = lanl.aip(positive_class,X,Y,negative_class)
        auc_pred6[i].append(roc_auc_score(y,x))

## Plot each AUC as a line to show that the variance of the points decreases (or calculate the variance directly and plot the estimates)
for j in range(len(l)):
    plt.plot(j * np.ones(n_iter),auc_pred6[l[j]],'o-')

plt.xlabel('Size Multiplier')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using AIP via different size_multiplier')
plt.show()




#weighted AIP
auc_pred7 = {}#dictionary
weight_idx = [1,0.99,0.98,0.97]

#compute wieght
weight_store = {}#dictionary
for w in weight_idx:
   weight_store[w] =  np.array([ w **(56-t) for t in range(56)])
   auc_pred7[w] = [] 

l=10
for n in range(l):
    print('\rIteration: ', str(n+1), ' / ', str(l), sep='', end='')
    positive_class,negative_class = lanl.resampling2(A_observed,A[56],size_multiplier = 1)
    for w in weight_idx:
        x,y = lanl.aip(positive_class,X,Y,negative_class,weight_store[w])
        auc_pred7[w].append(roc_auc_score(y,x))
    
    
for j in range(len(weight_idx)):
    plt.plot(np.linspace(1,l,l),auc_pred7[weight_idx[j]],'o-')
#plt.legend(['unweighted','weighted 0.99^(56-t)','weighted 0.98^(56-t)','weighted 0.97^(56-t)','weighted 0.96^(56-t)','weighted 0.95^(56-t)'])
plt.legend(['1','0.99','0.98','0.97','0.96','0.95'])
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score using weighted AIP')
plt.show()




#more weighting for previous Sunday if want to predict for Sunday
non_week_weight = np.array([ 0.99 **(56-t) for t in range(56)])
#multiple parameter for corresponding day
#week_idx = np.array([1.05,1.1,1,1.2,1.4,1.6])
week_idx = np.array([1,1.01,1.02,1.03,1.04,1.05])
week_weight_param = np.ones((len(week_idx),56))
for i in range(56):
    if (i+1)%7==1:
        week_weight_param[:,i] = week_idx

week_weight = {} #dictionary
for k in range(len(week_idx)):
    week_weight[week_idx[k]] = week_weight_param[k,:] * non_week_weight

auc_pred15 = {}
#auc_pred15[0]=[] #store AUC of AIP 
l=5
for w in week_idx:
    auc_pred15[w]=[]
for n in range(l):
    print('\rIteration: ', str(n+1), ' / ', str(l), sep='', end='')
    positive_class,negative_class = lanl.resampling2(A_observed,A[56],size_multiplier = 1)
    #compute AUC for unweighted AIP
    x1,y1=lanl.aip(positive_class,X,Y,negative_class)
    #auc_pred15[0].append(roc_auc_score(y1,x1))
    #for weighted AIP
    for w in week_idx:
        x2,y2 = lanl.aip(positive_class,X,Y,negative_class,week_weight[w])
        auc_pred15[w].append(roc_auc_score(y2,x2))

#plot of AUC curves
for j in range(len(week_idx)):
    plt.plot(np.linspace(1,l,l),auc_pred15[week_idx[j]],'o-')
#plt.plot(np.linspace(1,l,l),auc_pred15[0],'o-')
    
#plt.legend(['unweighted','weighted 0.99^(56-t)','weighted 0.98^(56-t)','weighted 0.97^(56-t)','weighted 0.96^(56-t)','weighted 0.95^(56-t)'])
#plt.legend(['1','1.05','1.1','1.2','1.4','1.6','unweighted'])
plt.legend(['1','1.01','1.02','1.03','1.04','1.05'])
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using weekly weighted AIP')
plt.show()





#linear/arithmetic weights
linear_model  = LinearRegression()
x = np.linspace(1,56,56).reshape(-1, 1)
#find linear regression of geometric weight of 0.99
y1 = non_week_weight.reshape(-1, 1) /np.sum(non_week_weight)
linear_model = linear_model.fit(x,y1)
pred_y1 = linear_model.predict(x)

auc_pred16 = {}
for k in range(3):
    auc_pred16[k]=[]
l=10
for i in range(l):
    print('\rIteration: ', str(i+1), ' / ', str(l), sep='', end='')
    positive_class,negative_class = lanl.resampling2(A_observed,A[56],size_multiplier = 1)
    lin_x1,lin_y1=lanl.aip(positive_class,X,Y,negative_class,y1)
    auc_pred16[0].append(roc_auc_score(lin_y1,lin_x1))
    lin_x2,lin_y2=lanl.aip(positive_class,X,Y,negative_class,pred_y1)
    auc_pred16[1].append(roc_auc_score(lin_y2,lin_x2))
    lin_x3,lin_y3=lanl.aip(positive_class,X,Y,negative_class,week_weight[1.05])
    auc_pred16[2].append(roc_auc_score(lin_y3,lin_x3))    

for j in range(3):
    plt.plot(np.linspace(1,l,l),auc_pred16[j],'o-')
plt.title('The AUC score of userdip data using linear weighted AIP')
plt.legend(['geometric','linear','weekly'])
plt.show()




#COSIE
#compute R_t using MASE for directed graph
hat_X,hat_Y,R = lanl.mase_direct(A_mat,22)

#compute R_average
R_average = lanl.average_mat(R,0,55)


#compute auc for prediction of A at t=57, repeat 100 times
auc_pred3= []
l=100
for i in range(l):
    print('\rIteration: ', str(i+1), ' / ', str(l), sep='', end='')
    positive_class,negative_class = lanl.resampling2(A_observed,A[56],size_multiplier = 2)
    x_cosie,y_cosie = lanl.cosie_average(positive_class,hat_X,hat_Y,R_average,negative_class)
    auc_pred3.append(roc_auc_score(y_cosie,x_cosie))

plt.plot(np.linspace(1,l,l),auc_pred3,'o-')
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using COSIE')
plt.show()


#repeating the resampling for size_multiplier = 0.1,0.2,0.5,1,2
auc_pred4 = {} ## dictionary
l = [0.1,0.2,0.5,1,2] ## list of resampling sizes
n_iter = 30 ## number of iterations per resampling size
## For each resampling size
for i in l:
    ## The 'value' of the dictionary for the 
    auc_pred4[i] = []
    ## Repeat for n_iter iterations
    for _ in range(n_iter):
        print('\rSize multiplier: ', str(i), '\tIteration: ', str(_+1), ' / ', str(n_iter), sep='', end='')
        positive_class,negative_class = lanl.resampling2(A_observed,A[56],size_multiplier = i)
        x_cosie,y_cosie = lanl.cosie_average(positive_class,hat_X,hat_Y,R_average,negative_class)
        auc_pred4[i].append(roc_auc_score(y_cosie,x_cosie))

## Plot each AUC as a line to show that the variance of the points decreases (or calculate the variance directly and plot the estimates)
for j in range(len(l)):
    plt.plot(j * np.ones(n_iter),auc_pred4[l[j]],'o-')

plt.xlabel('Size Multiplier')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using COSIE via different size_multiplier')
plt.show()


auc_pred8 = []
auc_pred9 =[]
auc_pred10 =[]
for t in range(56,90):
    print('\rIteration: ', str(t+1), ' / ', str(90), sep='', end='')
    positive_class,negative_class = lanl.resampling2(A_observed,A[t],size_multiplier = 2)
    #method1
    x_cosie,y_cosie = lanl.cosie_average(positive_class,hat_X,hat_Y,R_average,negative_class)
    auc_pred8.append(roc_auc_score(y_cosie,x_cosie))
    #method2
    R_average2 = lanl.average_mat(R,t-55,t)
    x_cosie,y_cosie = lanl.cosie_average(positive_class,hat_X,hat_Y,R_average2,negative_class)
    auc_pred9.append(roc_auc_score(y_cosie,x_cosie))
    #method3
    x_cosie,y_cosie = lanl.cosie_average(positive_class,hat_X,hat_Y,R[t],negative_class)
    auc_pred10.append(roc_auc_score(y_cosie,x_cosie))


#comparision of approaches
plt.plot(np.linspace(56,90,90-56),auc_pred8,'o-')
plt.plot(np.linspace(56,90,90-56),auc_pred9,'o-')
plt.plot(np.linspace(56,90,90-56),auc_pred10,'o-')
plt.legend(['Method 1','Method 2','Method 3'])
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('Comparision of COSIE approaches')
plt.show()



#comparison of T=7,10 and 56
#method 2
auc_pred11 = {}
period_T = [7,10,56]
for T in period_T:
    auc_pred11[T]=[]
for t in range(56,90):
    print('\rIteration: ', str(t+1), ' / ', str(90), sep='', end='')
    positive_class,negative_class = lanl.resampling2(A_observed,A[t],size_multiplier = 2)
    for T in period_T:
        R_average2 = lanl.average_mat(R,t-T+1,t)
        x_cosie,y_cosie = lanl.cosie_average(positive_class,hat_X,hat_Y,R_average2,negative_class)
        auc_pred11[T].append(roc_auc_score(y_cosie,x_cosie))

for j in range(len(period_T)):
    plt.plot(np.linspace(56,90,90-56),auc_pred11[period_T[j]],'o-')
plt.legend(['T=7','T=10','T=56'])
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data for different T')
plt.show()    



#compare of three    

auc_pred15 = {}
for g in range(3):
    auc_pred15[g]=[]
    
l=20
for i in range(l):
    print('\rIteration: ', str(i+1), ' / ', str(l), sep='', end='')
    positive_class,negative_class = lanl.resampling2(A_observed,A[56],size_multiplier = 1)
    #AIP
    x1,y1 = lanl.aip(positive_class,X,Y,negative_class)
    auc_pred15[0].append(roc_auc_score(y1,x1))
    #weighted AIP
    x2,y2 = lanl.aip(positive_class,X,Y,negative_class,week_weight[1.05])
    auc_pred15[1].append(roc_auc_score(y2,x2))
    #COSIE
    x_cosie,y_cosie = lanl.cosie_average(positive_class,hat_X,hat_Y,R_average,negative_class)
    auc_pred15[2].append(roc_auc_score(y_cosie,x_cosie))
    
for t in range(3):
    plt.plot(np.linspace(1,l,l),auc_pred15[t])

plt.legend(['AIP','Weighted AIP','COSIE'])  
plt.xlabel('Iterations')
plt.ylabel('AUC Scores')
plt.title('AUC Scores for three methods in prdiction of never observed links')
plt.show()



########################################################
# Compare expectations of probability by AIP and COSIE #
########################################################
auc_compare={}
auc_compare[0]=[]
auc_compare[1]=[]
for t in range(55):
    print('\rIteration: ', str(t+1), ' / ', str(55), sep='', end='')
    negative_class = lanl.resampling(A[t],size_multiplier = 2)
    x1,y1 = lanl.aip(A[t],X,Y,negative_class)
    x_cosie,y_cosie = lanl.cosie_average(A[t],hat_X,hat_Y,R[t],negative_class)
    auc_compare[0].append(roc_auc_score(y1,x1))
    auc_compare[1].append(roc_auc_score(y_cosie,x_cosie))


plt.figure(figsize=(8,4))
plt.plot(np.linspace(1,55,55),auc_compare[0],':o')
plt.plot(np.linspace(1,55,55),auc_compare[1],':o')
plt.legend(['AIP','COSIE'])
plt.xlabel('days')
plt.ylabel('AUC score')
plt.show()

