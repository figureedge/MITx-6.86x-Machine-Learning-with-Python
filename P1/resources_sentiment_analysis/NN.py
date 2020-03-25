
import project1 as p1
import utils
import numpy as np


X = np.array([[0, 1],[4,5]])
Y = np.array([[1,1], [2,2], [3,3]])

Z = X[:,np.newaxis]+Y

print(X)
print(Y)

print(Z.shape)
print(Z)