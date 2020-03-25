import numpy as np

### Functions for you to fill in ###

# pragma: coderesponse template


def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    # operator ** is the power of each element in matrix 
    # np.linalg.matrix_power is multiple matrix multiplication 
    kernel_matrix = (np.matmul(X, np.transpose(Y)) + c)**p

    return kernel_matrix

# pragma: coderesponse end

# pragma: coderesponse template


def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE


    ## sol 1 - broadcasting - an alternative sol for np.outer 
    # X (n, d)
    # X[:,np.newaxis] = X[:,np.newaxis,:] (n, 1, d)
    # Y (m, d)
    # broadcasting (n, m, d)
    # sum dim in d -> axis=2
    #  

    return np.exp(-gamma*(np.square(X[:,np.newaxis]-Y).sum(axis=2)))


    ## sol 2 - extract necessary elements from matrix multiplication 

    # xx = np.matmul(X, np.transpose(X))
    # xy = np.matmul(X, np.transpose(Y))   
    # yy = np.matmul(Y, np.transpose(Y))

    # dist = np.transpose([np.diag(xx)]) -2*xy + np.diag(yy)

    # return np.exp(-gamma*dist)


    ## sol 3 - iterate over each entity by nested loop 

    # [n,d] = np.shape(X)
    # [m,d] = np.shape(Y)
    # K = np.zeros((n,m))

    # for i in range(n):
    #     for j in range(m):
    #         dist = np.linalg.norm(X[i][:] - Y[j][:])
    #         print(dist)
    #         K[i][j] = np.exp(-gamma*np.power(dist,2))
    # return K


# pragma: coderesponse end





