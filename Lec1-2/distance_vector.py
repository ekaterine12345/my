import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance

a = np.array([1, 3])
b = np.array([2, 4])
distance1 = norm(b-a)
print(distance1)

distance1 = np.sqrt(np.sum((a-b)**2))
print(distance1)

distance1 = distance.euclidean(a, b)
print(distance1)
