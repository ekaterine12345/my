import numpy as np
a = np.array([[1, 3], [2, 4]])
print(a[:, 1])
print(np.linalg.det(a))
I = np.identity(5)
print(I)
dig = np.diag([10, 100, 600])
print(dig)
a_inv = np.linalg.inv(a)
print(a_inv)

# a            1, 3,  2,    4
b = np.array([[5, 5], [10, 32]])
print(a*b)
prod = np.dot(a, b)
print(prod)
print(a+b)
print(a.T)
print(np.transpose(a))
print(np.linalg.matrix_rank(a))

print(np.dot(a_inv, b))  # matrix equatino
print(a.trace())



