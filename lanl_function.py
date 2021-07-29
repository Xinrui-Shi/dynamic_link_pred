#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 18:10:34 2021

@author: shining
"""

import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from scipy.sparse.linalg import svds




#################################
# Constructing Adjacency Matrix #
#################################

#given threshold, construct a sequence adjacency matrices
def counter_A(data,multiple = None,direct=True):
    #convert edge information into list
    edge_array= np.array(data)

    #no threshold, one adjacency matrix
    if multiple is None:
        A = Counter()
        for link in edge_array:
            if direct:
                A[link[0],link[1]] = 1
            else:
                A[link[0],link[1]] = 1
                A[link[1],link[0]] = 1
        return A

    #mutiple adjacency matrix
    A_t ={}
    time_series = np.unique(edge_array[:,2])
    
    for t in time_series:
        A_t[t-np.min(time_series)] = Counter()
        
    for link in edge_array:
        if direct:
            A_t[link[2]-np.min(time_series)][link[0],link[1]] = 1
        else:
            A_t[link[2]-np.min(time_series)][link[0],link[1]] = 1
            A_t[link[2]-np.min(time_series)][link[1],link[0]] = 1
    return A_t
                       

#find common shape of adjacency matrix
def find_dimension(data):
    edge_array= np.array(data)
    m = np.max(edge_array[:,0])-np.min(edge_array[:,0])+1
    n = np.max(edge_array[:,1])-np.min(edge_array[:,1])+1
    return m,n
    


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


##############
# Resampling #
##############
#resampling to have same number of 0 and 1
def resampling(A,size_multiplier = 2):
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



############################
# Random Dot Product Graph #
############################
def compute_score(A_mat,B_mat,negative_class,positive_class): 
        
    ## Calculate the scores for negative_class
    scores_negative_class = []
    for pair in negative_class:
        scores_negative_class += [np.dot(A_mat[int(pair[0]),:],B_mat[:,int(pair[1])]) ]
        
    ## Calculate the scores for positive_class
    scores_positive_class = []
    for pair in positive_class:
        scores_positive_class += [np.dot(A_mat[int(pair[0]),:],B_mat[:,int(pair[1])]) ]
        
    #combine for x and y
    x = np.concatenate((np.array(scores_negative_class),np.array(scores_positive_class)))
    y = np.concatenate((np.zeros(len(scores_negative_class)),np.ones(len(scores_positive_class))))
    
    return x,y

def rdpg(A,U,V,negative_class,return_label=True):
    ## Calculate the scores for negative_class
    scores_negative_class = []
    ### %% FSP: Why negative_class[:n_edges,:]? It should simply be negative_class.
    ## for pair in negative_class[:n_edges,:]:
    for pair in negative_class:
        scores_negative_class += [np.dot(U[int(pair[0]),:],V[:,int(pair[1])]) ]
        
    ## Calculate the scores for positive_class
    scores_positive_class = []
    for pair in A:
        scores_positive_class += [np.dot(U[int(pair[0]),:],V[:,int(pair[1])]) ]
    #combine for x and y
    x = np.concatenate((np.array(scores_negative_class),np.array(scores_positive_class)))
    if return_label:
        y = np.concatenate((np.zeros(len(scores_negative_class)),np.ones(len(scores_positive_class))))
        return x,y
    else: 
        return x


######################
# other edge feature #
######################
#use hadamard to construct edge feature
def ef_hadamard(A,U,V,negative_class):

    ## Calculate the scores for negative_class
    scores_negative_class = []
    for pair in negative_class:
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

    ## Calculate the scores for negative_class
    scores_negative_class = []
    for pair in negative_class:
        scores_negative_class += [(U[int(pair[0]),:]+V[:,int(pair[1])])*0.5]
    ## Calculate the scores for positive_class
    scores_positive_class = []
    for pair in A:
        scores_positive_class += [(U[int(pair[0]),:]+V[:,int(pair[1])])*0.5]
    #combine for x and y
    x = np.vstack((np.array(scores_negative_class),np.array(scores_positive_class)))
    y = np.hstack((np.zeros(len(scores_negative_class)),np.ones(len(scores_positive_class))))
    return x,y

##############
# classifier #
##############

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


########
# DASE #
########
def dase(A,d):
    U,eigval,V_T = svds(A,k=d)
    sqrt_eigval = np.diag(np.sqrt(abs(eigval)))
    X = U @ sqrt_eigval
    Y = V_T.T @ sqrt_eigval
    return X,Y

#######
# AIP #
#######
def aip(A,X,Y,negative_class,weight=None):
    # X_t and Y_t are dictionary
    # A is counter
    t = len(X)
    
    if weight is None:
        weight = np.ones(t)/t
    else:
        weight = weight/np.sum(weight)
    
    for i in range(t):
        if i==0:
            x_sum,y = rdpg(A,X[i],Y[i].T,negative_class)
        else:
            x = rdpg(A,X[i],Y[i].T,negative_class,return_label=False)  
            x_sum += weight[i]*x
            
    return x_sum, y


    


########
# MASE #
########
def mase_direct(A_t,d):
    #apply DASE
    for t in range(len(A_t)):
        U,eigval,V_T = svds(A_t[t],k=d)
        hat_X = U
        hat_Y = V_T.T
        if t==0:
            til_X = hat_X
            til_Y = hat_Y
        else:
            til_X = np.hstack((til_X,hat_X))
            til_Y = np.hstack((til_Y,hat_Y))
    
    #apply SVD 
    U_X,eig_X,V_X = svds(til_X,k=d)
    U_Y,eig_Y,V_Y = svds(til_Y,k=d)
    
    #compute R_t
    R_t = {}
    for t in range(len(A_t)):
        R_t[t] = U_X.T @ A_t[t]@  U_Y
        
    return U_X,U_Y,R_t

#########
# COSIE #
#########
def average_mat(A,starting_idx=None,end_idx=None):
    if starting_idx is None:
            starting_idx = 0
    if end_idx is None:
            end_idx = len(A)-1
            
    for i in range(starting_idx,end_idx+1):
        if i==starting_idx:
            A_average = A[starting_idx]
        else:
            A_average += A[i] 
            
    return A_average/(end_idx+1-starting_idx)


def cosie_average(A_pred,X,Y,R_average,negative_class):
    XR = X @ R_average
    x,y = rdpg(A_pred,XR,Y.T,negative_class)
    
    return x,y


########
# UASE #   
########

#sorted index of destination node
def modify_counter(A,return_idx_list=False):
    #A is a counter
    A_keys = list(A.keys())
    #unmodified destination node index in order
    destination_node = sorted(np.unique([link[1] for link in A_keys]))
    
    
    #modfied counters
    A_modified = Counter()
    for link in A_keys:
        A_modified[link[0],destination_node.index(link[1])]=1
    
    if return_idx_list:
        return A_modified,destination_node
    else:
        return A_modified
    

def unfold_A(A,return_shape=False):
    #find shape of all adjacency matrices
    # construct unfolded A
    A_shape = {}#dictionary
    idx_list = list(A.keys())
    for i in idx_list:
        A_i = A[i]
        A_shape[i] = A_i.shape
        if i==idx_list[0]:
            unfolded_A = A_i.toarray()
        else:
            unfolded_A = np.hstack((unfolded_A,A_i.toarray()))
    if return_shape:
        return coo_matrix(unfolded_A),A_shape
    else:
        return coo_matrix(unfolded_A)

def uase(A,d):
    #A is a dictionary contain k adjacency matrices
    #d is dimension for truncated SVD
    
    #find shape of all adjacency matrices
    # construct unfolded A
    unfolded_A,A_shape = unfold_A(A,return_shape=True)
    #truncated SVD on dimension d
    U,eigval,V_transpose = svds(unfolded_A,k=d)
    
    #common X
    X = U @ np.diag(np.sqrt(abs(eigval)))
    
    #big Y
    Y = V_transpose.T @ np.diag(np.sqrt(abs(eigval)))
    #split Y
    Y_t = {}#dictionary
    #compute R_t
   # R_t = {}#dictionary
    #compute expected of A_t
    P_t = {}#dictionary
    idx_list = list(A.keys())
    n=0
    for idx in idx_list:
        n_i = A_shape[idx][1]
        Y_t[idx] = Y[n:(n_i+n),:]
       # R_t[idx] = X.T @ A[idx] @ Y_t[idx] 
        #P_t[idx] = X @ R_t[idx] @ Y_t[idx].T
        P_t[idx] = X  @ Y_t[idx].T
        n = n_i+n
    return P_t


#split snapshot for edge_array
def destination_node(data):
    edge_array = np.array(data)
    t_array = edge_array[:,2]
    split_idx = []
    time_idx = np.unique(t_array)
    j=0
    for i in range(len(t_array)-1):
        if len(split_idx)==len(time_idx)-1:
            break
        if t_array[i]==time_idx[j]:
            if t_array[i+1]==time_idx[j+1]:
                j+=1
                split_idx.append(i+1)
            
    edge_list = {}#dictionary
    t1 = time_idx[0] #first time t
    t2 = time_idx[len(time_idx)-1]#last time t
    
    edge_list[0] = edge_array[:split_idx[0]]
    edge_list[t2-t1] = edge_array[split_idx[t2-2]:]
    for k in range(1,t2-1):
        edge_list[k] = edge_array[split_idx[k-1]:split_idx[k]]
        
    #set of destination node
    V = {}
    for t in range(t2):
        V[t] = set(edge_list[t][:,1])
    return V

#construct V_til
def V_tilda(V):
    #combine all sets as V_til
    V_til = set([])
    time_idx = list(V.keys())
    for t in time_idx:
        V_til = V_til.union(V[t])
    return V_til

    
#construct Z_it's
def construct_Z(V,V_til=None):
    #V is dictionary contains sets of destination node
    if V_til is None:
        V_til = V_tilda(V)
    time_idx = list(V.keys())
    n_dest = np.max(np.array(list(V_til)))+1
    Z = np.zeros((n_dest,len(time_idx)))
    for j in list(V_til):
        for i in range(len(time_idx)):
            Z[j,i] += int(j in V[time_idx[i]])
    return Z


def z_it(A_counter,i):
    #A_counter is a Counter
    #i is destination node
    A_keys = np.array(list(A_counter.keys()))
    for link in A_keys:
        if link[1]==i:
            return 1
    return 0
        
#UASE AIP score
def uase_aip(A,Z,dest_idx_order,P_t,k,i):
    #A is disctionary contained counters (unmodified)
    #Z is contain all z_it
    #dest_idx_order is the destination node index in order
    #P_t is expectation
    #k is source node
    #i is destination node
    if np.sum(Z[i,:])==0:
        return 0
    #compute z_it and pick corresponding probabilities
    P_ki = np.array([])
    idx_list = list(A.keys())#time index
    for idx in idx_list:
        z = Z[i,idx]
        if z==1:
            p_kit = P_t[idx][k,dest_idx_order[idx].index(i)]
            P_ki = np.append(P_ki,p_kit)
    return np.average(P_ki)
   
########################
# Beta-bernoulli Model #
########################
def Beta_bernoulli(Z,i,T,alpha=1,beta=1):
    #Z is contain all z_it
    #i is destination node
    #T is time for prediction
    Z_i = Z[i,:]
    theta_i = np.array([])
    for t in range(T):
        if t!=0:
            alpha += Z_i[t+1]
            beta += 1- Z_i[t+1]
            
        theta_i = np.append(theta_i ,alpha/(alpha+beta))
    return theta_i
    











     

#########################################
# Temporal logistic regression approach #   
#########################################  

def design_matrix(Z,T,k):
    #Z is contain all z_it
    #T>=0 is time index
    m = Z.shape[0] # m is number of dstination nodes

    #construct of design matrix
    design_M = np.ones((m,k+1))
    design_M[:,1:] = Z[:,T-k:T]
  
    return design_M


def response_vector(Z,T):
    #construct of response vector
    response_v = Z[:,T]
    return response_v


def logit_matrix(Z,T,k):
    #Z is contain all z_it
    #T is latest time of observed adjacency matrix
    #k is number of past times for regression    
            
    #construct of design matrix
    design_M = design_matrix(Z,T,k)

    #construct of response vector
    response_v = response_vector(Z,T)
    
    return design_M,response_v



        
            
            
        
    
   
    
    



    








        
        







