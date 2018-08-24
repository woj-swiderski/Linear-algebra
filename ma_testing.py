import unittest
import matrix_algebra as ma

import numpy as np

# F = np.array([[1,2,3], [-2,0,0], [0,1,2]])

def F(vector):
    x, y, z = vector
    return ma.Vector([2*x-y, x+z, -x+y+3*z])


e1 = ma.Vector([1,0,0])
e2 = ma.Vector([0,1,0])
e3 = ma.Vector([0,0,1])

zero = ma.Vector([0,0,0])

v1 = -e1
v2 = -2*e2
v3 = v1 + v2

v = ma.Vector([1,2,3])

class Test_proj(unittest.TestCase):
    # test projection on basis vectors
    def test_standard(self):
        self.assertEqual(ma.proj(v, e1).all(), e1.all())
        self.assertEqual(ma.proj(v, e2).all(), (2*e2).all())
        self.assertEqual(ma.proj(v, e3).all(), (3*e3).all())
        self.assertEqual(ma.proj(v, v1).all(), (-e1).all())
        self.assertEqual(ma.proj(v, v2).all(), (-2*e2).all())
        self.assertEqual(ma.proj(v, v3).all(), (ma.proj(v, v1)+ma.proj(v,v2)).all())
        self.assertEqual(ma.proj(zero, v3).all(), zero.all())

        self.assertEqual(ma.proj(e1, e2).all(), zero.all())
        self.assertEqual(ma.proj(e3, v3).all(), zero.all())
        # self.assertEqual(ma.proj(v, v3).all(), (e1.all())

    # test projection on itself
    def test_itself(self):
        self.assertEqual(ma.proj(v, v).all(), v.all())
        self.assertEqual(ma.proj(v3, v3).all(), v3.all())
        self.assertEqual(ma.proj(v2, v2).all(), v2.all())


E = np.eye(3)   # standard basis in R^3
B = np.array([[1,2,3], [0,-2,1], [-1,1,2]]).T
v4 = 4*B[:,0] - 2*B[:,1] + B[:,2]   # so decomposition is [4,-2,1]


class Test_decomposition(unittest.TestCase):
    # decomposition in standard basis
    def test_standard(self):
        self.assertAlmostEqual(ma.decompose_in_basis(e1, E).all(), e1.all())
        self.assertAlmostEqual(ma.decompose_in_basis(-e2, E).all(), (-e2).all()) # because of type difference
        self.assertAlmostEqual(ma.decompose_in_basis(v4, E).all(), v4.all())
        self.assertAlmostEqual(ma.decompose_in_basis(v3, E).all(), v3.all())

    def test_nonstandard(self):
        self.assertAlmostEqual(ma.decompose_in_basis(v4, B).all(), np.array([4,-2,1]).all())


class Test_compose_decompose(unittest.TestCase):
    # test reversality compose <-> decompose
    def test_compose_decompose(self):
        c4 = ma.decompose_in_basis(v4, B)
        self.assertEqual(v4.all(), ma.compose_in_basis(c4, B).all())
        self.assertEqual(ma.decompose_in_basis(ma.compose_in_basis([1,2,3], B), B).all(), np.array([1,2,3]).all())

        a = ma.decompose_in_basis(ma.compose_in_basis([-2,0,3], B), B)
        self.assertAlmostEqual(np.linalg.norm(a-ma.Vector([-2,0,3])), 0)

        a = ma.decompose_in_basis(ma.compose_in_basis([2,-2,5], B), B)
        self.assertAlmostEqual(np.linalg.norm(a-ma.Vector([2,-2,5])), 0)



"""


e1, e2, e3 = Vector([1,3,-2]), Vector([0,1,2]), Vector([0,-3,1])

E = np.hstack((e1, e2, e3))    # basis as 2d-array

# matrix of F in basis E
MF = np.hstack((decompose_in_basis(F(e1), E), decompose_in_basis(F(e2), E),
                decompose_in_basis(F(e3), E)))
               
x = Vector([2,1,-3])
xE = decompose_in_basis(x, E)  # xE - set of coeffs of x in E

t = MF@xE  # set of coeffs of resulting F(xE) in basis E

# print(F(x))
# compose_in_basis(t, E)


# ### Let's assume we have two bases
# Let ${\bf E, E_1}$ be two basis: ${\bf E} = [e_1, e_2,\dots, e_n]$, ${\bf E_1} = [e_1^{'}, e_2^{'},\dots, e_n^{'}]$.
# 
# Let $M$ be a two-dimensional array with **rows** built as follows: the first row is composed of coeffs of decomposition of $e_1^{'}$ in ${\bf E}$, the second row is composed of coeffs of decomposition of $e_2$ in ${\bf E}$ and so on.
# 
# Similarly, let $M_1$ be a two-dimensional array arising from decomposition of vectors of ${\bf E}$ in ${\bf E_1}$. One can easily verify that $M$ and $M_1$ are *reciprocally inverse*.
# 
# Now, if $[x_1,x_2,\dots,x_n]$ are coeffs of $x$ in ${\bf E}$ and $[x_1^{'},x_2^{'},\dots,x_n^{'}]$ are coeffs of $x$ in ${\bf E_1}$ then one can also check that
# 
# $[x_1^{'},x_2^{'},\dots,x_n^{'}] = [x_1,x_2,\dots,x_n]\circ M^{-1} = [x_1,x_2,\dots,x_n]\circ M_1$ (cf. above).

# In[95]:


E = np.array([[1,2,3], [1,0,1], [0,-1,2]]).T
E1 = np.array([[1,1,1], [-1,-1,2], [1,2,0]]).T

M = np.vstack([decompose_in_basis(E1[:,j], E).reshape(1,-1) for j in range(3)])  # decompose vectors from E1 in E
M1 = np.vstack([decompose_in_basis(E[:,j], E1).reshape(1,-1) for j in range(3)])

# print(np.hstack([sum(M[k,j]*E[:,j] for j in range(3)).reshape(-1,1) for k in range(3)])) # recompose E1 from M


# In[99]:


x = Vector([7,-1,3])
xE = decompose_in_basis(x, E)
xE1 = decompose_in_basis(x, E1)

print(xE)
print(xE1)


# In[98]:


print(xE1.T@M)

"""