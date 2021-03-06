# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
#import functions from files
import lanl_function as lanl
import zhu

# load data as pandas data frames
data_userdip= pd.read_csv('userdip.txt',header=None)
#data_usersip= pd.read_csv('usersip.txt',header=None)


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

plt.figure(figsize=(8,4))
plt.plot(np.linspace(1,l,l),auc_pred,'o:')
plt.xlabel('Iterations')
plt.ylabel('AUC score')
#plt.title('The AUC score of userdip data using RGPD (using A_train)')
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

plt.figure(figsize=(8,4))
## Plot each AUC as a line to show that the variance of the points decreases (or calculate the variance directly and plot the estimates)
for j in range(len(l)):
    plt.plot(j * np.ones(n_iter),auc_pred2[l[j]],'o:')

plt.xlabel('Size Multiplier')
plt.ylabel('AUC score')
#plt.title('The AUC score using RGPD via different size_multiplier')
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
    
plt.figure(figsize=(8,4))
plt.plot(np.linspace(1,l,l),auc_pred5,'o:')
plt.xlabel('Iterations')
plt.ylabel('AUC score')
#plt.title('The AUC score of userdip data using AIP on Multiple Adjacency Matrice')
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
        
plt.figure(figsize=(8,4))
## Plot each AUC as a line to show that the variance of the points decreases (or calculate the variance directly and plot the estimates)
for j in range(len(l)):
    plt.plot(j * np.ones(n_iter),auc_pred6[l[j]],'o:')

plt.xlabel('Size Multiplier')
plt.ylabel('AUC score')
#plt.title('The AUC score of userdip data using AIP via different size_multiplier')
plt.show()


################
# weighted AIP #
################
auc_pred7 = {}#dictionary
weight_idx = [1,0.99,0.98,0.97]

#compute wieght
weight_store = {}#dictionary
for w in weight_idx:
   weight_store[w] =  np.array([ w **(56-t) for t in range(56)])
   auc_pred7[w] = [] 

l=20
for n in range(l):
    print('\rIteration: ', str(n+1), ' / ', str(l), sep='', end='')
    negative_class = lanl.resampling(A[56],size_multiplier = 1)
    for w in weight_idx:
        x,y = lanl.aip(A[56],X,Y,negative_class,weight_store[w])
        auc_pred7[w].append(roc_auc_score(y,x))
    
plt.figure(figsize=(7,4))
for j in range(len(weight_idx)):
    plt.plot(np.linspace(1,l,l),auc_pred7[weight_idx[j]],'o:')
#plt.legend(['unweighted','weighted 0.99^(56-t)','weighted 0.98^(56-t)','weighted 0.97^(56-t)','weighted 0.96^(56-t)','weighted 0.95^(56-t)'])
plt.legend(['w=1','w=0.99','w=0.98','w=0.97'])
plt.xlabel('Iterations')
plt.ylabel('AUC score')
#plt.title('The AUC score of userdip data using weighted AIP')
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
#week_idx = np.array([1.05,1.1,1,1.2,1.4,1.6])
week_idx = np.array([1,1.03,1.05,1.07,1.1])
#week_idx = np.array([1,1.05,1.1])
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
plt.figure(figsize=(7,4))
for j in range(len(week_idx)):
    plt.plot(np.linspace(1,l,l),auc_pred15[week_idx[j]],'o:')
plt.plot(np.linspace(1,l,l),auc_pred15[0],'o:')
    
#plt.legend(['unweighted','weighted 0.99^(56-t)','weighted 0.98^(56-t)','weighted 0.97^(56-t)','weighted 0.96^(56-t)','weighted 0.95^(56-t)'])
#plt.legend(['1','1.05','1.1','1.2','1.4','1.6','unweighted'])
#plt.legend(['1','1.01','1.02','1.03','1.04','1.05'])
#plt.legend(['b=1','b=1.05','b=1.1','unweighted'])
plt.legend(['b=1','b=1.03','b=1.05','b=1.07','b=1.1','unweighted'])
plt.xlabel('Iterations')
plt.ylabel('AUC score')
plt.ylim(0.99275,0.99303)
#plt.title('The AUC score of userdip data using weekly weighted AIP')
plt.show()


#plot of weekly weights
for w in week_idx:
    plt.plot(np.linspace(1,56,56),week_weight[w]/np.sum(week_weight[w]))
plt.hlines(xmin=1,xmax=56,y=1/56)
plt.legend(['1','1.05','1.1','1.2','1.4','1.6','unweighted'])
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
    lin_x3,lin_y3=lanl.aip(A[56],X,Y,negative_class,week_weight[1.03])
    auc_pred16[2].append(roc_auc_score(lin_y3,lin_x3))    

for j in range(3):
    plt.plot(np.linspace(1,l,l),auc_pred16[j],'o:')
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
plt.ylabel('weights')
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

plt.figure(figsize=(8,4))
plt.plot(np.linspace(1,l,l),auc_pred3,'o:')
plt.xlabel('Interations')
plt.ylabel('AUC score')
#plt.title('The AUC score of userdip data using COSIE')
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

plt.figure(figsize=(8,4))
## Plot each AUC as a line to show that the variance of the points decreases (or calculate the variance directly and plot the estimates)
for j in range(len(l)):
    plt.plot(j * np.ones(n_iter),auc_pred4[l[j]],'o:')

plt.xlabel('Size Multiplier')
plt.ylabel('AUC score')
#plt.title('The AUC score of userdip data using COSIE via different size_multiplier')
plt.show()


#prediction for t=57,58,....90

#using first 56 adjacency matrices
auc_pred8 = []
for t in range(56,90):
    print('\rIteration: ', str(t+1), ' / ', str(90), sep='', end='')
    negative_class = lanl.resampling(A[t],size_multiplier = 2)
    x_cosie,y_cosie = lanl.cosie_average(A[t],hat_X,hat_Y,R_average,negative_class)
    auc_pred8.append(roc_auc_score(y_cosie,x_cosie))

plt.figure(figsize=(8,4))
plt.plot(np.linspace(56,90,90-56),auc_pred8,'o:')
plt.xlabel('days')
plt.ylabel('AUC score')
#plt.title('The AUC score of userdip data using COSIE for t=57,..90')
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
plt.figure(figsize=(8,4))
plt.plot(np.linspace(56,90,90-56),auc_pred8,'o:')
plt.plot(np.linspace(56,90,90-56),auc_pred9,'o:')
plt.plot(np.linspace(56,90,90-56),auc_pred10,'o:')
plt.legend(['Method 1','Method 2','Method 3'])
plt.xlabel('Days')
plt.ylabel('AUC score')
#plt.title('Comparision of approaches')
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
plt.figure(figsize=(8,4))
for j in range(len(period_T)):
    plt.plot(np.linspace(56,90,90-56),auc_pred11[period_T[j]],'o:')
plt.legend(['T=7','T=10','T=56'])
plt.xlabel('Days')
plt.ylabel('AUC score')
#plt.title('The AUC score of userdip data for different T')
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


#compare the two methods using AIP
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


auc_pred12a= []
auc_pred13a= []
auc_pred12b= []
auc_pred13b= []
l=10
for i in range(l):
    print('\rIteration: ', str(i+1), ' / ', str(l), sep='', end='')
    negative_class = lanl.resampling(A[56],size_multiplier = 1)
    #COSIE
    #resampling method1
    x1,y1= lanl.cosie_average(A[56],hat_X,hat_Y,R_average,negative_class)
    auc_pred12a.append(roc_auc_score(y1,x1))
    #resampling method2
    negative_class = np.vstack((negative_class,type2_pair))
    x2,y2= lanl.cosie_average(A[56],hat_X,hat_Y,R_average,negative_class)
    auc_pred13a.append(roc_auc_score(y2,x2))
    #AIP
    #resampling method1
    x1,y1= lanl.aip(A[56],X,Y,negative_class)
    auc_pred12b.append(roc_auc_score(y1,x1))
    #resampling method2
    negative_class = np.vstack((negative_class,type2_pair))
    x2,y2= lanl.aip(A[56],X,Y,negative_class)
    auc_pred13b.append(roc_auc_score(y2,x2))

plt.figure(figsize=(7,4))
plt.plot(np.linspace(1,l,l),auc_pred12a,'o--',color='#808080')
plt.plot(np.linspace(1,l,l),auc_pred13a,'o:',color='#808080')
plt.plot(np.linspace(1,l,l),auc_pred12b,'o--',color='#00008B')
plt.plot(np.linspace(1,l,l),auc_pred13b,'o:',color='#00008B')
plt.legend(['COSIE:Method1','COSIE:Method2','AIP:Method1','AIP:Method2'])
plt.xlabel('Iterations')
plt.ylabel('AUC score')
plt.show()

plt.figure(figsize=(7,1))
plt.plot(np.linspace(1,l,l),auc_pred12a,'o--',color='#808080')
plt.xlabel('Iterations')
plt.ylabel('AUC score')
plt.show()


plt.figure(figsize=(12,8))
ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2,rowspan=2) 
ax1.plot(np.linspace(1,l,l),auc_pred12a,'o--',color='#808080')
ax1.plot(np.linspace(1,l,l),auc_pred13a,'o:',color='#808080')
ax1.plot(np.linspace(1,l,l),auc_pred12b,'o--',color='#00008B')
ax1.plot(np.linspace(1,l,l),auc_pred13b,'o:',color='#00008B')
ax1.legend(['COSIE:Method1','COSIE:Method2','AIP:Method1','AIP:Method2'])
ax1.set_xlabel('Iterations')
ax1.set_xlabel('AUC score')
ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2,rowspan=1) 
ax2.plot(np.linspace(1,l,l),auc_pred12a,'o--',color='#808080')
ax3 = plt.subplot2grid((3, 4), (1, 2), colspan=2,rowspan=1)
ax3.plot(np.linspace(1,l,l),auc_pred13a,'o:',color='#808080')
ax4 = plt.subplot2grid((3, 4), (2, 0), colspan=2,rowspan=1)
ax4.plot(np.linspace(1,l,l),auc_pred12b,'o--',color='#00008B')
ax5 = plt.subplot2grid((3, 4), (2, 2), colspan=2,rowspan=1)
ax5.plot(np.linspace(1,l,l),auc_pred13b,'o:',color='#00008B')
plt.show()


plt.figure(figsize=(12,8))
ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4,rowspan=2) 
ax1.plot(np.linspace(1,l,l),auc_pred12a,'o--',color='#808080')
ax1.plot(np.linspace(1,l,l),auc_pred13a,'o:',color='#808080')
ax1.plot(np.linspace(1,l,l),auc_pred12b,'o--',color='#00008B')
ax1.plot(np.linspace(1,l,l),auc_pred13b,'o:',color='#00008B')
ax1.legend(['COSIE:Method1','COSIE:Method2','AIP:Method1','AIP:Method2'])
ax1.set_xlabel('Iterations')
ax1.set_xlabel('AUC score')
ax2 = plt.subplot2grid((4, 4), (2, 0), colspan=1,rowspan=1) 
ax2.plot(np.linspace(1,l,l),auc_pred12a,'o--',color='#808080')
ax3 = plt.subplot2grid((4, 4), (2, 1), colspan=1,rowspan=1)
ax3.plot(np.linspace(1,l,l),auc_pred13a,'o:',color='#808080')
ax4 = plt.subplot2grid((4, 4), (2, 2), colspan=1,rowspan=1)
ax4.plot(np.linspace(1,l,l),auc_pred12b,'o--',color='#00008B')
ax5 = plt.subplot2grid((4, 4), (2, 3), colspan=1,rowspan=1)
ax5.plot(np.linspace(1,l,l),auc_pred13b,'o:',color='#00008B')
plt.show()


#########
# MRDPG #
#########

#construct adjacency matrix for time 1 to 90 days
A = lanl.counter_A(data_userdip,multiple = True,direct=True)
m,n = lanl.find_dimension(data_userdip) #find dimension of adjacency matrix
#modified index of destination node
A_modified = {}#dictionary
modified_idx = {}
for i in range(len(A)):
    A_mod_counter,modified_idx_list = lanl.modify_counter(A[i],return_idx_list=True)
    A_modified[i] = A_mod_counter
    modified_idx[i] = modified_idx_list
    
A_mat = {}
#same number of source node
for i in range(len(A)):
    A_mat[i] = lanl.counter2A(A_modified[i],m,n=None)


#########################
#Estimation of theta_it #
#########################

#find all sets of destination node
V = lanl.destination_node(data_userdip)

#find all sets of source node
V_source = lanl.destination_node(data_userdip,return_dest=False)

#Bernoulli_beta model

#construct Z_it's
Z = lanl.construct_Z(V)

#compute theta
theta = {}
T=56
l = 100
import random
random_selected_node = random.sample(range(0, Z.shape[0]), l)
for i in range(l):
    print('\rIteration: ', str(i+1), ' / ', str(l), sep='', end='')
    theta[i]=lanl.Beta_bernoulli(Z,random_selected_node[i],T,alpha=1,beta=1)

for i in range(l):
    plt.plot(np.linspace(0,T,T+1),theta[i])
plt.ylabel('estimate of theta_i')
plt.xlabel('t')
plt.title('Estimate of theta for destination nodes')
plt.show()

#explore different initial params (alpha,beta)
    
#different alpha
beta0 =1
beta_list2 = []
    
i=0
theta_i = {}
for alpha0 in range(1,10):
    beta_list2.append((alpha0,beta0))
    theta_i[alpha0]=lanl.Beta_bernoulli(Z,i,T,alpha0,beta0)
    plt.plot(np.linspace(1,T,T),theta_i[alpha0])
plt.legend(beta_list2)
plt.ylabel('estimate of theta_i')
plt.xlabel('t')
plt.title('Estimate of theta for destination node i=0')
plt.show()

#different beta
alpha0 =1
beta_list3 = []
    
i=0
theta_i = {}
for beta0 in range(1,10):
    beta_list3.append((alpha0,beta0))
    theta_i[beta0]=lanl.Beta_bernoulli(Z,i,T,alpha0,beta0)
    plt.plot(np.linspace(1,T,T),theta_i[beta0])
plt.legend(beta_list3)
plt.ylabel('estimate of theta_i')
plt.xlabel('t')
plt.title('Estimate of theta for destination node i=0')
plt.show()


#logistic

#different k

k=88
theta2 = np.zeros((Z.shape[0],k))
for k in range(1,k+1):
    print('\rIteration: ', str(k), ' / ', str(88), sep='', end='')
    M1,V1 = lanl.logit_matrix(Z,89,k)
    logit_model =  LogisticRegression(max_iter=5000)
    logit_model.fit(M1,V1)
    beta1 = logit_model.coef_
    pred_V =  np.dot(M1,beta1.T)
    exp_z = np.exp(pred_V)
    theta_hat = exp_z /(1+exp_z)
    theta2[:,k-1] = theta_hat.reshape(len(theta_hat),)
    
for i in range(10):
    plt.plot(np.linspace(1,88,88),theta2[i])
plt.ylabel('estimate of theta_i')
plt.xlabel('k')
plt.title('Estimate of theta for destination nodes')
plt.show()

#for given k, compute theta_i along time
k=7
theta3 = np.zeros((Z.shape[0],90-k))
for t in range(k,90):
    print('\rIteration: ', str(t), ' / ', str(89), sep='', end='')
    M1,V1 = lanl.logit_matrix(Z,t,k)
    logit_model =  LogisticRegression(max_iter=5000)
    logit_model.fit(M1,V1)
    beta1 = logit_model.coef_
    pred_V =  np.dot(M1,beta1.T)
    exp_z = np.exp(pred_V)
    theta_hat = exp_z /(1+exp_z)
    theta3[:,t-k] = theta_hat.reshape(len(theta_hat),) 

    
for i in range(10):
    plt.plot(np.linspace(k,90,90-k),theta3[i,:],'-o')
plt.ylabel('estimate of theta_i')
plt.xlabel('t')
plt.title('Estimate of theta for destination nodes')
plt.show()   





#test for several A_ts as observed pairs
#A1,...A7:T=7
train_data = data_userdip.iloc[:1299164,:]

#construct adjacency matrix for time 1 to 56  days
A_train = lanl.counter_A(train_data,multiple = True,direct=True)
m,n = lanl.find_dimension(train_data) #find dimension of adjacency matrix
#modified index of destination node
A_modified_train = {}#dictionary
modified_idx_train = {}
for i in range(len(A_train)):
    A_mod_counter,modified_idx_list = lanl.modify_counter(A_train[i],return_idx_list=True)
    A_modified_train[i] = A_mod_counter
    modified_idx_train[i] = modified_idx_list
    
A_mat_train = {}
#same number of source node
for i in range(len(A_train)):
    A_mat_train[i] = lanl.counter2A(A_modified_train[i],m,n=None)

#find all sets of destination node
V_train = lanl.destination_node(train_data)
#construct Z_it'sA
Z_train = lanl.construct_Z(V_train)

#count for all observed source node
source_node_count = lanl.count_node_number(train_data.iloc[:,0])

#compute P_t
P_t =  lanl.uase(A_mat_train,20)#d=20 for truncated svd

#compute theta_i
T=56
Theta = lanl.Beta_bernoulli(Z_train,T,alpha0=1,beta0=1)
theta = Theta[:,Theta.shape[1]-1]

positive_class = np.zeros((len(A[56]),2))
i=0
for pair in A[56]:
    positive_class[i,:] = pair
    i+=1

#two types of estimates for theta_i


#construct design matrix
design_matrix1 = lanl.design_matrix(Z_train,T,55)
y= Z_train[:,Z_train.shape[1]-1]
## Fit logistic regression
logreg = LogisticRegression(random_state=0).fit(design_matrix1, y)
## Predicted probabilities
design_matrix2 = lanl.design_matrix(Z_train[:,1:],55,55)
theta_i = logreg.predict_proba(design_matrix2)[:,1]


auc_pred14 = {}
for num in range(4):
    auc_pred14[num]=[]

ll = 10
for l in range(ll):
    print('\rIteration: ', str(l+1), ' / ', str(ll), sep='', end='')
    negative_class = lanl.resampling(A[56],size_multiplier = 0.5)
    observed_link= np.vstack((positive_class,negative_class))
    observed_v = np.concatenate((np.ones(len(positive_class)),np.zeros(len(negative_class))))
    
    prob1 = np.array([])
    prob2 = np.array([])
    prob3 = np.array([])
    prob4 = np.array([])
    for p in range(len(observed_link)):
        k = int(observed_link[p,0])
        i = int(observed_link[p,1])
        phi_ik1 = lanl.uase_aip(A_modified_train,Z_train,modified_idx_train,source_node_count,P_t,k,i,m)
        phi_ik2 = lanl.uase_aip(A_modified_train,Z_train,modified_idx_train,source_node_count,P_t,k,i,m,simple_est=True)
        prob1 = np.append(prob1,theta_i[i]*phi_ik1) #logit
        prob2 = np.append(prob2,theta[i]*phi_ik1) #beta_bernoulli
        prob3 = np.append(prob3,theta_i[i]*phi_ik2) #logit
        prob4 = np.append(prob4,theta[i]*phi_ik2) #beta_bernoulli
        
    auc_pred14[0].append(roc_auc_score(observed_v,prob1))
    auc_pred14[1].append(roc_auc_score(observed_v,prob2))
    auc_pred14[2].append(roc_auc_score(observed_v,prob3))
    auc_pred14[3].append(roc_auc_score(observed_v,prob4))

plt.figure(figsize=(8,4))    
for num in range(4):
    plt.plot(np.linspace(1,ll,ll),auc_pred14[num],':o')
#plt.legend(['Logit,ratio estimate','beta-bern,ratio estimate','logit,simple estimate','beta-bern,simple estimate'])
plt.legend(['1,1','2,1','1,2','2,2'])
plt.ylabel('AUC score')
plt.xlabel('iterations')
#plt.title('AUC curves for different combinations of estimations')
plt.show()

        










