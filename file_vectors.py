import numpy as np
from numpy import inf
import math
from scipy.spatial import distance

a = np.array([1, 2, 3])
# n1 = np.linalg.norm(a)
# n2 = np.linalg.norm(a, 1)
# n3 = np.count_nonzero(a)
# n_inf = np.linalg.norm(a, inf)
# print(a)
# print(n1)
# print(n2)
# print(n3)
# print(n_inf)

b = np.array([2, 4, 5])
product = np.dot(a, b)
print(product)
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
cost = product / (norm_a * norm_b)
print(cost)
print(math.degrees(math.acos(cost)))
print(a - b)

dist1 = np.linalg.norm(a-b)
print(dist1)

print(np.sqrt(np.sum((a-b)**2)))

dist = distance.euclidean(a, b)
print(dist)