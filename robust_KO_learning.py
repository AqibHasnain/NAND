import cvxpy
from cvxpy import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


def calc_Koopman(Yf,Yp,flag=1,lambda_val=0.0,noise_scaler=1,lambda_val_1=0.0,verbose=True):
    
    ngenes = Yf.shape[0]
    ndatapts = Yf.shape[1]
    solver_instance = cvxpy.SCS
    
    if flag == 1: # least-squares solution
        Yp_inv = np.linalg.pinv(Yp)
        K = np.dot(Yf,Yp_inv)
        # TO DO: Add SVD based DMD for modal decompostion (modes can be insightful)
        
    if flag == 2: # LASSO optimization 
        print('L1 Regularization')
        operator = Variable(shape=(ngenes,ngenes))
        
        print("[INFO]: CVXPY Koopman operator variable: " + repr(operator.shape))
        print("[INFO]: Shape of Yf and Yp: " + repr(Yf.shape) + ', ' + repr(Yp.shape) )
        
        if type(lambda_val) == float:
            reg_term = lambda_val * cvxpy.norm(operator,p=1)
        else: 
            Unoise = np.tile(lambda_val,ndatapts)
            reg_term = cvxpy.norm(cvxpy.matmul(operator + np.eye(ngenes),noise_scaler*Unoise),p=1)
        norm2_fit_term = cvxpy.norm(Yf - cvxpy.matmul(operator,Yp),p=2)
        
        objective = Minimize(norm2_fit_term + reg_term)
        constraints = []
        prob = Problem(objective,constraints)
        result = prob.solve(verbose=verbose,solver=solver_instance,max_iters=int(1e6))
        K = operator.value
        
        print("[INFO]: CVXPY problem status: " + prob.status)
        
    if flag == 3: # robust optimization approach
        print("L2,2 Regularization")
        
        operator = Variable(shape=(ngenes,ngenes)) # Koopman matrix  K
        
        print("[INFO]: CVXPY Koopman operator variable: " + repr(operator.shape))
        print("[INFO]: Shape of Yf and Yp: " + repr(Yf.shape) + ', ' + repr(Yp.shape) )
        
        if type(lambda_val) == float:
            reg_term = lambda_val * cvxpy.norm(operator,p='fro')
        else: 
            Unoise = np.tile(lambda_val,ndatapts)
            reg_term = cvxpy.norm(cvxpy.matmul(operator + np.eye(ngenes),noise_scaler*Unoise),p='fro') # in 2019, we had assumed that 
        norm2_fit_term = cvxpy.norm(Yf - cvxpy.matmul(operator,Yp),p=2)
        
        objective = Minimize(norm2_fit_term + reg_term)
        constraints = []
        prob = Problem(objective,constraints)
        result = prob.solve(verbose=verbose,solver=solver_instance,max_iters=int(1e6))
        K = operator.value
        
        print("[INFO]: CVXPY problem status: " + prob.status)
    
    if flag == 4: 
        print('L1 and L2,2 Regularization')
        operator = Variable(shape=(ngenes,ngenes)) # Koopman matrix  K
        
        print("[INFO]: CVXPY Koopman operator variable: " + repr(operator.shape))
        print("[INFO]: Shape of Yf and Yp: " + repr(Yf.shape) + ', ' + repr(Yp.shape) )
        
        Unoise = np.tile(lambda_val,ndatapts)
        regF_term = cvxpy.norm(cvxpy.matmul(operator + np.eye(ngenes),noise_scaler*Unoise),p='fro') 
        reg1_term = lambda_val_1 * cvxpy.norm(operator,p=1)

        norm2_fit_term = cvxpy.norm(Yf - cvxpy.matmul(operator,Yp),p=2)
        
        objective = Minimize(norm2_fit_term + regF_term + reg1_term)
        constraints = []
        prob = Problem(objective,constraints)
        result = prob.solve(verbose=verbose,solver=solver_instance,max_iters=20000)
        K = operator.value
        
        print("[INFO]: CVXPY problem status: " + prob.status)
        
    if flag == 5: # robust optimization approach
        print("L2,2 (robust) Regularization but exactly as theory suggests, so only noise from Yp should be used")
        
        operator = Variable(shape=(ngenes,ngenes)) # Koopman matrix  K
        
        print("[INFO]: CVXPY Koopman operator variable: " + repr(operator.shape))
        print("[INFO]: Shape of Yf and Yp: " + repr(Yf.shape) + ', ' + repr(Yp.shape) )
        
        if type(lambda_val) == float:
            reg_term = lambda_val * cvxpy.norm(operator,p='fro')
        else: 
            Unoise = np.tile(lambda_val,ndatapts)
            reg_term = cvxpy.norm(cvxpy.matmul(operator + np.eye(ngenes),noise_scaler*Unoise),p='fro') 
        norm2_fit_term = cvxpy.norm(Yf - cvxpy.matmul(operator,Yp),p=2)
        
        objective = Minimize(norm2_fit_term + reg_term)
        constraints = []
        prob = Problem(objective,constraints)
        result = prob.solve(verbose=verbose,solver=solver_instance,max_iters=int(1e6))
        K = operator.value
        
        print("[INFO]: CVXPY problem status: " + prob.status)

                
    print('MSE =  ' + '{:0.3e}'.format(mean_squared_error(Yf,K@Yp))) 
    print('\n','\n')

    return K

def calc_input_Koopman(LHS,Up,flag=1,lambda_val=0.0,noise_scaler=1,verbose=True):
    ''' From Yf=AYp+BUp or Yf-AYp=BUp (A known) or Yf-(AYp + B_1U1 + ... )=B_JUJ (B_1 to B_J-1 known), we see 
        that the optimization can always be written as ||loss_lhs - operator*Up||_2 + ||G(operator)||_F.
    '''
    ngenes = LHS.shape[0]
    ndatapts = LHS.shape[1]
    ninputs = Up.shape[0]

    if flag == 1: # least-squares solution
        Up_inv = np.linalg.pinv(Up)
        Ki = np.dot(LHS,Up_inv)
        print('The mean squared error is: ' + '{:0.3e}'.format(np.linalg.norm(LHS - Ki@Up)**2 / ndatapts)) 
        # TO DO: Add SVD based DMD for modal decompostion (modes can be insightful)
        
    if flag == 2: # robust optimization approach
        solver_instance = cvxpy.SCS
        operator = Variable(shape=(ngenes,ninputs)) # Koopman matrix  K
        
        print("[INFO]: CVXPY Koopman operator variable: " + repr(operator.shape))
        
        if type(lambda_val) == float:
            reg_term = lambda_val * cvxpy.norm(operator,p=1) 
        else:
            Unoise = np.tile(lambda_val,ndatapts)
            reg_term = cvxpy.norm(cvxpy.matmul(operator,noise_scaler*Unoise),p='fro') # where exactly does this term come from? 

        norm2_fit_term = cvxpy.norm(LHS - cvxpy.matmul(operator,Up),p=2) 
        objective = Minimize(norm2_fit_term + reg_term)
        constraints = []
        prob = Problem(objective,constraints)
        result = prob.solve(verbose=verbose,solver=solver_instance,max_iters=20000)
        Ki = operator.value

        print("[INFO]: CVXPY problem status: " + prob.status)
        print('MSE =  ' + '{:0.3e}'.format(mean_squared_error(LHS,Ki@Up))) 
        print('\n','\n')

    return Ki

def calc_circuit_and_host_impact(Yf,Yp,Khh,circuit_inds,lambda_val=0.0,noise_scaler=1,verbose=True):
    
    ngenes = Yp.shape[0]
    ndatapts = Yp.shape[1]
    ncircuitgenes = len(circuit_inds)
    
    solver_instance = cvxpy.SCS
    
    Khc = Variable(shape=(ngenes-ncircuitgenes,ncircuitgenes)) # circuit impact on host
    Kch = Variable(shape=(ncircuitgenes,ngenes-ncircuitgenes)) # retroactivity (host impact on circuit)
#     Kch = np.zeros((ncircuitgenes,ngenes-ncircuitgenes))
    Kcc = Variable(shape=(ncircuitgenes,ncircuitgenes))
    lowerK = cvxpy.hstack((Khc,Khh))
    upperK = cvxpy.hstack((Kcc,Kch))
    operator = cvxpy.vstack((upperK,lowerK))

    print("[INFO]: CVXPY Khc variable: " + repr(Khc.shape))
    print("[INFO]: CVXPY Kch variable: " + repr(Kch.shape))
    print("[INFO]: CVXPY Kcc variable: " + repr(Kcc.shape))
    print("[INFO]: CVXPY operator variable: " + repr(operator.shape))

    if type(lambda_val) == float:
        reg_term = lambda_val*( cvxpy.norm(Khc,p=1) + cvxpy.norm(Kch,p=1) + cvxpy.norm(Kcc,p=1))
#         reg_term = lambda_val*( cvxpy.norm(Kch,p=1) + cvxpy.norm(Kch,p='fro') )
    else:
        lambda_val = lambda_val[:,np.newaxis]
        Unoise = np.tile(lambda_val,ndatapts)
        reg_term = cvxpy.norm(cvxpy.matmul(operator,noise_scaler*Unoise),p='fro') 
    
    norm2_fit_term = cvxpy.norm(Yf - cvxpy.matmul(operator,Yp), p=2)
    objective = Minimize(norm2_fit_term + reg_term)
    constraints = []
    prob = Problem(objective,constraints)
    result = prob.solve(verbose=verbose,solver=solver_instance,max_iters=int(1e6))
    Kstar = operator.value

    print("[INFO]: CVXPY problem status: " + prob.status)
    print('MSE =  ' + '{:0.3e}'.format(np.linalg.norm(Yf - Kstar@Yp)**2 / ndatapts)) 
    print('\n','\n')

    return Kstar

def koopman_heatmap(thisK,figsize,xlabels,ylabels,shrink,savefig=False,savedir=''):
    my_cmap = sns.diverging_palette(15, 221, s=99, sep=1, l=45, center='light',as_cmap=True)
    fig = plt.figure(figsize=figsize)
    hm = sns.heatmap(thisK,linewidths=0.0,cmap=my_cmap,xticklabels=xlabels,yticklabels=ylabels,
                linecolor='white',square=True,annot=False,annot_kws={"size":11},fmt='2.1f',
                     cbar_kws={'shrink':shrink},center=0)
    if thisK.shape[1] == 1: 
        plt.xticks(fontsize=20,rotation='horizontal');
    else: 
        plt.xticks(fontsize=20,rotation='vertical');
    plt.yticks(fontsize=18,rotation='horizontal')
    plt.axhline(y=0, color='k',linewidth=3)
    plt.axhline(y=thisK.shape[0], color='k',linewidth=3)
    plt.axvline(x=0, color='k',linewidth=3)
    plt.axvline(x=thisK.shape[1], color='k',linewidth=3)
    cax = fig.axes[-1]
    cax.set_frame_on(True)
    for spine in cax.spines.values():
        spine.set(visible=True, lw=1, edgecolor='k')
    plt.tight_layout()
    if savefig:
        plt.savefig(savedir,dpi=300,bbox_inches='tight',transparent=True)
    plt.show()