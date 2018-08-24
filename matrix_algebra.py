"""
matrix_algebra provides some matrix-algebra utilities:
- orthogonal projection of vector x on y
- Gramm-Schmidt orthogonalisation of a basis (relative to standard dot product)
- decomposition of a vector in give basis
- composition of a vector from its coefficients (reversal to decomposition)
"""

import numpy as np


class Vector(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).reshape(-1,1)
        return obj


def proj(v, u):
    """
    Projects orthogonally vector v on u (vectors are of shapes (n,1))

    proj([2,3], [0,1]) = [0,3]
    """
    # what if u == 0?
    return (u.T@v) / (u.T@u) * u


def GS(V, inplace=False):
    """
    Gramm-Schmidt ortogonalization
    
    V - n x m matrix - hstack of m n-dimensional vectors (n x 1 arrays).
    returns: n x m matrix of orthogonalized vectors
    
    if inplace==True V is modified; otherwise a new matrix is returned
    """
    U = np.array(V, copy=True)  # equivalent to np.copy(V)
    m = V.shape[1]
    for j in range(m):
        for i in range(j):
            U[:,j] -= proj(V[:,j], U[:,i])
    return U


def decompose_in_basis(vector, basis):
    """
    vector - n x 1 array
    basis - n x n array thought of as hstack of n x 1 dimensional creating basis of R^n
    returns - coeffs of vector in basis 
    """
    return np.linalg.solve(basis, vector)


def compose_in_basis(coeffs, basis):
    """
    computes vector from its coeffs in basis
    """
    return Vector(basis@coeffs)


# ## Remarks on linear transformations
# In order to find (or build) the matrix of a linear transformation $F$ in a given basis ${\bf B} = \{e_1, e_2,\dots, e_n\}$ you must:
# * decompose images $F(e_i)$ of basis vectors in the basis ${\bf B}$
# * put sets of coefficients of that decomposition as consecutive columns of the matrix $M$.
# Then, if you have the matrix $M$ and a vector $v$ you can compute the value $F(v)$ as follows:
# * decompose $v$ in ${\bf B}$ to get coefficients $c_1, c_2,\dots, c_n$
# * compute `M @ c` where `c` is a vector $[c_1, c_2,\dots,c_n]$ - this gives you coeffs of $F(v)$ in ${\bf B}$
# * compute $F(v)$ using calculated coeffs and ${\bf B}$


if __name__ == "__main__":
    ...