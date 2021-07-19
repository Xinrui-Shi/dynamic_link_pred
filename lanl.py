# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
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
    print('\rIteration: ', str(i+1), ' / ', str(l), sep='', end='')
    negative_class_A_test = lanl.resampling(A_test,size_multiplier = 2)
    x,y=lanl.rdpg(A_test,U_train,V_train,negative_class_A_test)
    auc_pred.append(roc_auc_score(y,x))

plt.plot(np.linspace(1,l,l),auc_pred,'o-')
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using RGPD (using A_train)')
plt.show()

#repeating the resampling for size_multiplier = 0.1,0.2,0.5,1,2
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


#######
# AIP #
#######
X = {}
Y = {}
for t in range(56):
    X_t,Y_t = lanl.dase(A_mat[t],d=22)
    X[t] = X_t
    Y[t] = Y_t

#compute auc for prediction of A at t=57, repeat 100 times
auc_pred5= []
l=50
for i in range(l):
    print('\rIteration: ', str(i+1), ' / ', str(l), sep='', end='')
    negative_class = lanl.resampling(A[56],size_multiplier = 1)
    x,y = lanl.aip(A[56],X,Y,negative_class)
    auc_pred5.append(roc_auc_score(y,x))

plt.plot(np.linspace(1,l,l),auc_pred5,'o-')
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using AIP on Multiple Adjacency Matrice')
plt.show()

#repeating the resampling for size_multiplier = 0.1,0.2,0.5,1,2
auc_pred6 = {} ## dictionary
l = [0.1,0.2,0.5,1,2] ## list of resampling sizes
n_iter = 10 ## number of iterations per resampling size
## For each resampling size
for i in l:
    ## The 'value' of the dictionary for the 
    auc_pred6[i] = []
    ## Repeat for n_iter iterations
    for _ in range(n_iter):
        print('\rSize multiplier: ', str(i), '\tIteration: ', str(_+1), ' / ', str(n_iter), sep='', end='')
        negative_class = lanl.resampling(A[56],size_multiplier = i)
        x,y = lanl.aip(A[56],X,Y,negative_class)
        auc_pred6[i].append(roc_auc_score(y,x))

## Plot each AUC as a line to show that the variance of the points decreases (or calculate the variance directly and plot the estimates)
for j in range(len(l)):
    plt.plot(j * np.ones(n_iter),auc_pred6[l[j]],'o-')

plt.xlabel('Size Multiplier')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using AIP via different size_multiplier')
plt.show()


################
# weighted AIP #
################
auc_pred7 = {}#dictionary
weight_idx = [1,0.99,0.98,0.97,0.96,0.95]

#compute wieght
weight_store = {}#dictionary
for w in weight_idx:
   weight_store[w] =  np.array([ w **(56-t) for t in range(56)])
   auc_pred7[w] = [] 

l=50
for n in range(l):
    print('\rIteration: ', str(n+1), ' / ', str(l), sep='', end='')
    negative_class = lanl.resampling(A[56],size_multiplier = 1)
    for w in weight_idx:
        x,y = lanl.aip(A[56],X,Y,negative_class,weight_store[w])
        auc_pred7[w].append(roc_auc_score(y,x))
    
    
for j in range(len(weight_idx)):
    plt.plot(np.linspace(1,l,l),auc_pred7[weight_idx[j]],'o-')
#plt.legend(['unweighted','weighted 0.99^(56-t)','weighted 0.98^(56-t)','weighted 0.97^(56-t)','weighted 0.96^(56-t)','weighted 0.95^(56-t)'])
plt.legend(['1','0.99','0.98','0.97','0.96','0.95'])
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using weighted AIP')
plt.show()

#plot of geometric weights
for w in weight_idx:
   plt.plot(np.linspace(1,56,56),weight_store[w]/np.sum(weight_store[w]))
plt.legend(['1','0.99','0.98','0.97','0.96','0.95'])
plt.title('Plot of geometric weights')
plt.show()

#expontial weighting
auc_pred14 = {}#dictionary
exp_weight_idx = [0,0.1,0.2,0.3,0.4,0.5]
exp_weight = {}#dictionary
for w in exp_weight_idx:
   exp_weight[w] =  np.array([ np.exp(w*t) for t in range(56)])
   auc_pred14[w] = []
l=10
for n in range(l):
    print('\rIteration: ', str(n+1), ' / ', str(l), sep='', end='')
    negative_class = lanl.resampling(A[56],size_multiplier = 1)
    for w in exp_weight_idx:
        x,y = lanl.aip(A[56],X,Y,negative_class,exp_weight[w])
        auc_pred14[w].append(roc_auc_score(y,x))
        
for j in range(len(exp_weight_idx)):
    plt.plot(np.linspace(1,l,l),auc_pred14[exp_weight_idx[j]],'o-')
    
#plt.legend(['unweighted','weighted 0.99^(56-t)','weighted 0.98^(56-t)','weighted 0.97^(56-t)','weighted 0.96^(56-t)','weighted 0.95^(56-t)'])
plt.legend(['0','0.1','0.2','0.3','0.4','0.5'])
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using exponential weighted AIP')
plt.show()

#plot of exponential weights
for w in exp_weight_idx:
   plt.plot(np.linspace(1,56,56),exp_weight[w]/np.sum(exp_weight[w]))
plt.legend(['0','0.1','0.2','0.3','0.4','0.5'])
plt.title('Plot of expotential weights')
plt.show()




#more weighting for previous Sunday if want to predict for Sunday
non_week_weight = np.array([ 0.99 **(56-t) for t in range(56)])
#multiple parameter for corresponding day
week_idx = np.array([1,1.2,1.4,1.6,1.8,2.0])
week_weight_param = np.ones((len(week_idx),56))
for i in range(56):
    if (i+1)%7==1:
        week_weight_param[:,i] = week_idx

week_weight = {} #dictionary
for k in range(len(week_idx)):
    week_weight[week_idx[k]] = week_weight_param[k,:] * non_week_weight

auc_pred15 = {}
auc_pred15[0]=[] #store AUC of AIP 
l=10
for w in week_idx:
    auc_pred15[w]=[]
for n in range(l):
    print('\rIteration: ', str(n+1), ' / ', str(l), sep='', end='')
    negative_class = lanl.resampling(A[56],size_multiplier = 1)
    #compute AUC for unweighted AIP
    x1,y1=lanl.aip(A[56],X,Y,negative_class)
    auc_pred15[0].append(roc_auc_score(y1,x1))
    #for weighted AIP
    for w in week_idx:
        x2,y2 = lanl.aip(A[56],X,Y,negative_class,week_weight[w])
        auc_pred15[w].append(roc_auc_score(y2,x2))

#plot of AUC curves
for j in range(len(week_idx )):
    plt.plot(np.linspace(1,l,l),auc_pred15[week_idx[j]],'o-')
plt.plot(np.linspace(1,l,l),auc_pred15[0],'o-')
    
#plt.legend(['unweighted','weighted 0.99^(56-t)','weighted 0.98^(56-t)','weighted 0.97^(56-t)','weighted 0.96^(56-t)','weighted 0.95^(56-t)'])
plt.legend(['1','1.2','1.4','1.6','1.8','2.0','unweighted'])
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using weekly weighted AIP')
plt.show()


#plot of weekly weights
for w in week_idx:
    plt.plot(np.linspace(1,56,56),week_weight[w]/np.sum(week_weight[w]))
plt.hlines(xmin=1,xmax=56,y=1/56)
plt.legend(['1','1.2','1.4','1.6','1.8','2.0','unweighted'])
plt.title('Plot of weekly weights')
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
    negative_class = lanl.resampling(A[56],size_multiplier = 1)
    lin_x1,lin_y1=lanl.aip(A[56],X,Y,negative_class,y1)
    auc_pred16[0].append(roc_auc_score(lin_y1,lin_x1))
    lin_x2,lin_y2=lanl.aip(A[56],X,Y,negative_class,pred_y1)
    auc_pred16[1].append(roc_auc_score(lin_y2,lin_x2))
    lin_x3,lin_y3=lanl.aip(A[56],X,Y,negative_class,week_weight[1.2])
    auc_pred16[2].append(roc_auc_score(lin_y3,lin_x3))    

for j in range(3):
    plt.plot(np.linspace(1,l,l),auc_pred16[j],'o-')
plt.title('The AUC score of userdip data using linear weighted AIP')
plt.legend(['geometric','linear','weekly'])
plt.show()


#plot of four 

plt.plot(np.linspace(1,56,56),non_week_weight/np.sum(non_week_weight))
plt.plot(np.linspace(1,56,56),pred_y1/np.sum(pred_y1))
plt.plot(np.linspace(1,56,56),week_weight[1.2]/np.sum(week_weight[1.2]))
plt.hlines(xmin=1,xmax=56,y=1/56)
plt.legend(['geo~0.99','linear','weekly','unweighted'])
plt.xlabel('t')
plt.ylabel('AUC')
plt.title('Plot of Scaled Weights')
plt.show()

#########
# COSIE #
#########

#compute R_t using MASE for directed graph
hat_X,hat_Y,R = lanl.mase_direct(A_mat,22)

#compute R_average
R_average = lanl.average_mat(R,0,55)


#compute auc for prediction of A at t=57, repeat 100 times
auc_pred3= []
l=100
for i in range(l):
    print('\rIteration: ', str(i+1), ' / ', str(l), sep='', end='')
    negative_class = lanl.resampling(A[56],size_multiplier = 2)
    x_cosie,y_cosie = lanl.cosie_average(A[56],hat_X,hat_Y,R_average,negative_class)
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
        negative_class = lanl.resampling(A[56],size_multiplier = i)
        x_cosie,y_cosie = lanl.cosie_average(A[56],hat_X,hat_Y,R_average,negative_class)
        auc_pred4[i].append(roc_auc_score(y_cosie,x_cosie))

## Plot each AUC as a line to show that the variance of the points decreases (or calculate the variance directly and plot the estimates)
for j in range(len(l)):
    plt.plot(j * np.ones(n_iter),auc_pred4[l[j]],'o-')

plt.xlabel('Size Multiplier')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using COSIE via different size_multiplier')
plt.show()


#prediction for t=57,58,....90

#using first 56 adjacency matrices
auc_pred8 = []
for t in range(56,90):
    print('\rIteration: ', str(t+1), ' / ', str(90), sep='', end='')
    negative_class = lanl.resampling(A[t],size_multiplier = 2)
    x_cosie,y_cosie = lanl.cosie_average(A[t],hat_X,hat_Y,R_average,negative_class)
    auc_pred8.append(roc_auc_score(y_cosie,x_cosie))
plt.plot(np.linspace(56,90,90-56),auc_pred8,'o-')
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using COSIE for t=57,..90')
plt.show()


#using observed adjacency matrix t-55 to t-1 for prediction (average R)
auc_pred9 =[]
for t in range(56,90):
    print('\rIteration: ', str(t+1), ' / ', str(90), sep='', end='')
    negative_class = lanl.resampling(A[t],size_multiplier = 2)
    R_average2 = lanl.average_mat(R,t-55,t)
    x_cosie,y_cosie = lanl.cosie_average(A[t],hat_X,hat_Y,R_average2,negative_class)
    auc_pred9.append(roc_auc_score(y_cosie,x_cosie))
plt.plot(np.linspace(56,90,90-56),auc_pred9,'o-')
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using COSIE for t=57,..90')
plt.show()

#using observed adjacency matrix t-55 to t-1 for prediction (not average R)
auc_pred10 =[]
for t in range(56,90):
    print('\rIteration: ', str(t+1), ' / ', str(90), sep='', end='')
    negative_class = lanl.resampling(A[t],size_multiplier = 2)
    x_cosie,y_cosie = lanl.cosie_average(A[t],hat_X,hat_Y,R[t],negative_class)
    auc_pred10.append(roc_auc_score(y_cosie,x_cosie))
plt.plot(np.linspace(56,90,90-56),auc_pred10,'o-')
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using COSIE for t=57,..90')
plt.show()


auc_pred8 = []
auc_pred9 =[]
auc_pred10 =[]
for t in range(56,90):
    print('\rIteration: ', str(t+1), ' / ', str(90), sep='', end='')
    negative_class = lanl.resampling(A[t],size_multiplier = 2)
    #method1
    x_cosie,y_cosie = lanl.cosie_average(A[t],hat_X,hat_Y,R_average,negative_class)
    auc_pred8.append(roc_auc_score(y_cosie,x_cosie))
    #method2
    R_average2 = lanl.average_mat(R,t-55,t)
    x_cosie,y_cosie = lanl.cosie_average(A[t],hat_X,hat_Y,R_average2,negative_class)
    auc_pred9.append(roc_auc_score(y_cosie,x_cosie))
    #method3
    x_cosie,y_cosie = lanl.cosie_average(A[t],hat_X,hat_Y,R[t],negative_class)
    auc_pred10.append(roc_auc_score(y_cosie,x_cosie))


#comparision of approaches
plt.plot(np.linspace(56,90,90-56),auc_pred8,'o-')
plt.plot(np.linspace(56,90,90-56),auc_pred9,'o-')
plt.plot(np.linspace(56,90,90-56),auc_pred10,'o-')
plt.legend(['Method 1','Method 2','Method 3'])
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('Comparision of approaches')
plt.show()
    
    
    
#comparison of T=7,10 and 56
#method 2
auc_pred11 = {}
period_T = [7,10,56]
for T in period_T:
    auc_pred11[T]=[]
for t in range(56,90):
    print('\rIteration: ', str(t+1), ' / ', str(90), sep='', end='')
    negative_class = lanl.resampling(A[t],size_multiplier = 2)
    for T in period_T:
        R_average2 = lanl.average_mat(R,t-T+1,t)
        x_cosie,y_cosie = lanl.cosie_average(A[t],hat_X,hat_Y,R_average2,negative_class)
        auc_pred11[T].append(roc_auc_score(y_cosie,x_cosie))

for j in range(len(period_T)):
    plt.plot(np.linspace(56,90,90-56),auc_pred11[period_T[j]],'o-')
plt.legend(['T=7','T=10','T=56'])
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data for different T')
plt.show()    
     

#alternative resampling method
#combine all the linked pairs in T=1 to 56
A_first56 = Counter() #counter
for i in range(56):
    A_first56 += A[i]
#store all pair of them
type2_pair = np.zeros((len(A_first56),2))  
j=0  
for pair in A_first56:
    #print('\rIteration: ', str(j+1), ' / ', str(len(A_first56)), sep='', end='')
    type2_pair[j,0] = pair[0]
    type2_pair[j,1] = pair[1]
    j+=1
    
#previous link but not link at t= 57
selected_idx = np.array([pair not in A[56] for pair in A_first56])    
#selected all type1 pairs after deleting some linked pair at t=57
type2_pair = type2_pair[selected_idx]


#compare the two methods using COSIE
auc_pred12= []
auc_pred13= []
l=50
for i in range(l):
    print('\rIteration: ', str(i+1), ' / ', str(l), sep='', end='')
    negative_class = lanl.resampling(A[56],size_multiplier = 2)
    #resampling method1
    x1,y1= lanl.cosie_average(A[56],hat_X,hat_Y,R_average,negative_class)
    auc_pred12.append(roc_auc_score(y1,x1))
    #resampling method2
    negative_class = np.vstack((negative_class,type2_pair))
    x2,y2= lanl.cosie_average(A[56],hat_X,hat_Y,R_average,negative_class)
    auc_pred13.append(roc_auc_score(y2,x2))

plt.plot(np.linspace(1,l,l),auc_pred12,'o-')
plt.plot(np.linspace(1,l,l),auc_pred13,'o-')
plt.legend(['Method1','Method2'])
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using COSIE to compare resampling method')
plt.show()


#compare the two methods using COSIE
auc_pred12= []
auc_pred13= []
l=10
for i in range(l):
    print('\rIteration: ', str(i+1), ' / ', str(l), sep='', end='')
    negative_class = lanl.resampling(A[56],size_multiplier = 1)
    #resampling method1
    x1,y1= lanl.aip(A[56],X,Y,negative_class)
    auc_pred12.append(roc_auc_score(y1,x1))
    #resampling method2
    negative_class = np.vstack((negative_class,type2_pair))
    x2,y2= lanl.aip(A[56],X,Y,negative_class)
    auc_pred13.append(roc_auc_score(y2,x2))

plt.plot(np.linspace(1,l,l),auc_pred12,'o-')
plt.plot(np.linspace(1,l,l),auc_pred13,'o-')
plt.legend(['Method1','Method2'])
plt.xlabel('Times')
plt.ylabel('AUC score')
plt.title('The AUC score of userdip data using AIP to compare resampling method')
plt.show()









