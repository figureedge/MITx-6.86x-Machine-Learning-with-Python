"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    [n,d] = np.shape(X)
    K = len(mixture.p)

    NK = []

    for i in range(K):

        A = np.power(2*np.pi*mixture.var[i], -d/2) # (1,1) -> ()
        B = np.linalg.norm(X-mixture.mu[i], axis=1) # (n,1) -> (1,n) -> (n,)
        C = np.exp(-1/2/mixture.var[i]*B**2) # (n,1) -> (1,n) -> (n,)

        NK.append(A*C) # pdf for each class before mixture NK -> (K,n)

    N = np.matmul(mixture.p, NK) # mixture pdf # (1,n) -> (n,)
    L = np.sum(np.log(N))

    N_weight = NK*mixture.p.reshape(K,1) # weighting each class -> (K,n)
    N_post = N_weight/N # posterior is normalized pdf for each class with weighting -> (K,n)

    return np.transpose(N_post), L


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    [n,d] = np.shape(X)
    [n,K] = np.shape(post)

    n_K = np.sum(post, axis=0) # (1,K) -> (K,)
    p_K = n_K/n # (1,K) -> (K,)

    mu_K = np.matmul(post.transpose(), X) # (K,d)
    mu_K = mu_K.transpose()/n_K
    mu_K = mu_K.transpose()

    var_K = [] # list

    for i in range(K):
        A = np.linalg.norm(X-mu_K[i], axis=1) # (n,1) -> (n,)
        B = np.matmul(A**2, post[:,i])
        var_K.append(B/d/n_K[i]) # (1,K) -> (K,)

    var_K = np.asarray(var_K) # convert to array

    return GaussianMixture(mu_K, var_K, p_K)



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
    mixture = mstep(X, post)
    [post, L] = estep(X, mixture)
    
    while L-L0 > 1e-6*abs(L):

        mixture = mstep(X, post)
        [post, L_update] = estep(X, mixture)

        L0 = L
        L = L_update
        
    return mixture, post, L
