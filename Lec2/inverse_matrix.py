import numpy as np
a = np.array([[1, 2], [2, 1]])
a_inv = np.linalg.inv(a)
print(a_inv)

product = np.dot(a, a_inv)
print(product)
