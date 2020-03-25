import numpy as np
import em
import common

#############################

from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
import naive_em

###############################

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

# X = np.loadtxt("netflix_incomplete.txt")
# X_gold = np.loadtxt("netflix_complete.txt")


K = 4
n, d = X.shape
seed = 0

# TODO: Your code here


[mixture, post] = common.init(X,K,seed)

# [m,n] = np.shape(X)

# for i in range(m):

#     Cu = np.where(X[i] != 0)[0] # return tuple so need [0]
#     Hu = np.where(X[i] == 0)[0]

#     print(Cu)
#     print(Hu)

print(X)
print(mixture)
print(em.estep(X, mixture))

# print(naive_em.estep(X, mixture))
