import numpy as np
import kmeans
import common
import naive_em
import em

from matplotlib import pyplot as plt

from common import GaussianMixture



X = np.loadtxt("toy_data.txt")

# TODO: Your code here

########################### 2. K-means

# cost_array = np.zeros([5,4])
# cost_array.fill(np.nan) # allocate non array


# for seed in range(5):

#     for K in range(1,5):

#         [mixture, post] = common.init(X,K,seed)
#         [mixture, post, cost] = kmeans.run(X, mixture, post)
        
#         cost_array[seed, K-1] = cost
#         cost_min = np.min(cost_array, axis=0) # min for each column

#         # common.plot(X, mixture, post, 'clustering')

# print(cost_array)
# print(cost_min)


######################## 3. Expectation–maximization algorithm


# L_array = np.zeros([5,4])
# L_array.fill(np.nan) # allocate non array


# for seed in range(5):

#     for K in range(1,5):

#         [mixture, post] = common.init(X,K,seed)
#         [mixture, post, L] = naive_em.run(X, mixture, post)
        
#         L_array[seed, K-1] = L
#         L_min = np.min(L_array, axis=0) # min for each column

#         common.plot(X, mixture, post, 'clustering')

# print(L_array)
# print(L_min)

############################ 4. Comparing K-means and EM


# K = 3
# seed = 0

# [mixture, post] = common.init(X,K,seed)
# [mixture, post, cost] = kmeans.run(X, mixture, post)
# common.plot(X, mixture, post, 'clustering')

# print(mixture)

# [mixture, post] = common.init(X,K,seed)
# [mixture, post, L] = naive_em.run(X, mixture, post)
# common.plot(X, mixture, post, 'clustering')

# print(mixture)



######################### 5. Picking the best K
######################### 5. Bayesian Information Criterion

# BIC_array = []
# seed = 0

# for K in range(1,5):

#     [mixture, post] = common.init(X,K,seed)
#     [mixture, post, L] = naive_em.run(X, mixture, post)

#     BIC = common.bic(X, mixture, L)

#     BIC_array.append(BIC)


# BIC_best = max(BIC_array) # least penalty BIC
# K_best = BIC_array.index(max(BIC_array))+1 # nest K index

# print(BIC_best, K_best)

######################### 8. Using the mixture model for collaborative filtering

# X = np.loadtxt("netflix_incomplete.txt")
# X_gold = np.loadtxt('netflix_complete.txt')
 
X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt('test_complete.txt')

K = 4
seed = 0

[mixture, post] = common.init(X,K,seed)

[post, L] = em.estep(X, mixture)
mixture = em.mstep(X, post, mixture)
print(post)
print(L)
print(mixture)

[mixture, post, L] = em.run(X, mixture, post)
print(post)
print(L)
print(mixture)


X_prep = em.fill_matrix(X, mixture)
print(X_prep)

RMSE = common.rmse(X_gold, X_prep)
print(RMSE)


K = 4

for seed in range(5):

    [mixture, post] = common.init(X,K,seed)
    [mixture, post, L] = em.run(X, mixture, post)   
    print(L)

    X_prep = em.fill_matrix(X, mixture)
    RMSE = common.rmse(X_gold, X_prep)
    print(RMSE)
