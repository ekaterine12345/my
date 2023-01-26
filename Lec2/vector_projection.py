import numpy as np

u = np.array([1, 2])
v = np.array([3, 4])

dot_product = np.dot(u, v)
vnorm = np.linalg.norm(v) ** 2
projected = (dot_product / vnorm) * v
print(projected)
complement = u - projected
print(complement)
print(np.dot(projected, complement))


