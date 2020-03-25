"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    
    [n,d] = np.shape(X)
    k = len(mixture.p)
    NK = np.zeros([n,k])

    ########################### cartisian domain 

    # for i in range(n):
    #     Cu = np.where(X[i] != 0)[0] # return tuple so need [0]
    #     Hu = np.where(X[i] == 0)[0]
    #     d = len(Cu) # dimension decided by non-zero features 

    #     for j in range(k):

    #         A = np.power(2*np.pi*mixture.var[j], -d/2) # (1,1) -> ()        
    #         B = np.linalg.norm(X[i,Cu] - mixture.mu[j,Cu]) # (1,1) -> ()
    #         C = np.exp(-1/2/mixture.var[j]*B**2) # (1,1) -> ()

    #         # K-class Gaussian before mixture  
    #         NK[i,j] = A*C # (n,k)

    # # apply weighting to perform Gaussian mixture  
    # N_post = np.multiply(NK, mixture.p) # (n,k)
    # N_post_mix = np.sum(N_post, axis=1) # (n,1) -> (n,)

    # # log-likelihood
    # # normalized posterior 
    # L = np.sum(np.log(N_post_mix))
    # N_post_norm = N_post / N_post_mix[:,None]

    # return N_post_norm, L

    ############################ log domain

    for i in range(n):
        Cu = np.where(X[i] != 0)[0] # return tuple so need [0]
        Hu = np.where(X[i] == 0)[0]
        d = len(Cu) # dimension decided by non-zero features 

        for j in range(k):

            A = -d/2*np.log(2*np.pi*mixture.var[j]) # (1,1) -> ()        
            B = np.linalg.norm(X[i,Cu] - mixture.mu[j,Cu]) # (1,1) -> ()
            C = -1/2/mixture.var[j] * B**2 # (1,1) -> ()

            # K-class Gaussian before mixture  
            NK[i,j] = A+C # (n,k)

    # apply weighting to perform Gaussian mixture  
    N_post = NK + np.log(mixture.p) # (n,k)
    N_post_mix = logsumexp(N_post, axis=1) # (n,1) -> (n,)

    # log-likelihood
    # normalized posterior 
    L = np.sum(N_post_mix)
    N_post_norm = N_post - N_post_mix[:,None] 

    return np.exp(N_post_norm), L



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """

    [n,d] = np.shape(X)
    k = len(mixture.p)

    n_k = np.sum(post, axis=0) # (1,k) -> (k,)
    p_k = n_k/n # (1,k) -> (k,)

    # allocation array
    mu_k = np.zeros([k,d])
    var_k = np.zeros(k)

    # allocate list to append 
    non_zero_length = []

    # non-zero length for each sample 
    for i in range(n):

        non_zero_index = np.where(X[i] != 0)[0]
        non_zero_length.append(len(non_zero_index)) # list, (1,n)

    non_zero_length = np.asarray(non_zero_length) # (n,1) -> (n,)
    
    
    # mean estimation, (k,d) 
    for i in range(k):

        for j in range(d):
            index = np.where(X[:,j] != 0)[0] # index where X not zero
            
            # update condition
            if np.sum(post[index,i]) >= 1:
                mu_k[i,j] = np.inner(X[index,j], post[index,i]) / np.sum(post[index,i])
            else:
                mu_k[i,j] = mixture.mu[i,j]

    # var estimation, (1,k) -> (k,)
    B = np.zeros([n,k])

    for i in range(n):

        for j in range(k):
            index = np.where(X[i] != 0)[0] # index where X not zero

            A = np.linalg.norm(X[i,index] - mu_k[j,index])
            B[i,j] = post[i,j]*A**2

    var_k = np.sum(B, axis=0) / np.matmul(post.T, non_zero_length)
    
    # var criteria 
    index = np.where(var_k <= min_variance)[0]
    var_k[index] = min_variance

    return GaussianMixture(mu_k, var_k, p_k)

   

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """

   # initial value 
    [post, L0] = estep(X, mixture)
    mixture = mstep(X, post, mixture)
    [post, L] = estep(X, mixture)
    
    while L-L0 >= 1e-6*abs(L):

        mixture = mstep(X, post, mixture)
        [post, L_update] = estep(X, mixture)

        L0 = L
        L = L_update
        
    return mixture, post, L



def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    
    [n,d] = np.shape(X)
    k = len(mixture.p)
    NK = np.zeros([n,k])

    ################################# e-step ###########################
    # for i in range(n):
    #     Cu = np.where(X[i] != 0)[0] # return tuple so need [0]
    #     Hu = np.where(X[i] == 0)[0]
    #     d = len(Cu) # dimension decided by non-zero features 

    #     for j in range(k):

    #         A = -d/2*np.log(2*np.pi*mixture.var[j]) # (1,1) -> ()        
    #         B = np.linalg.norm(X[i,Cu] - mixture.mu[j,Cu]) # (1,1) -> ()
    #         C = -1/2/mixture.var[j] * B**2 # (1,1) -> ()

    #         # K-class Gaussian before mixture  
    #         NK[i,j] = A+C # (n,k)

    # # apply weighting to perform Gaussian mixture  
    # N_post = NK + np.log(mixture.p) # (n,k)
    # N_post_mix = logsumexp(N_post, axis=1) # (n,1) -> (n,)

    # # log-likelihood
    # # normalized posterior 
    # L = np.sum(N_post_mix)
    # N_post_norm = N_post - N_post_mix[:,None] 
    #
    # post = np.exp(N_post_norm)

    ##############################################
    [post, L] = estep(X, mixture)
    ##############################################

    # just a copy
    X_pred = np.copy(X)

    # expectation value 
    update = post @ mixture.mu # (n,d)

    # selection Hu
    for i in range(n):
        Cu = np.where(X[i] != 0)[0] # return tuple so need [0]
        Hu = np.where(X[i] == 0)[0]

        X_pred[i,Hu] = update[i,Hu]

    return X_pred
