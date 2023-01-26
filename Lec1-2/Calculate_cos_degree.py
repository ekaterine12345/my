import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

u = np.array([2, 2])
v = np.array([0, 3])
product = np.dot(u, v)

norm_u = np.linalg.norm(u)
norm_v = np.linalg.norm(v)

cost = product / (norm_u * norm_v)

print(cost)
print(math.degrees(math.acos(cost)))
print(np.rad2deg(np.arccos(cost)))

print('*'*100)
# print(u.ndim)
u = u.reshape(1, 2)  # ორგანზომილებიან არრაიში გადავიდა
v = v.reshape(1, 2)  # ორგანზომილებიან არრაიში გადავიდა
# print(u, v)
cost = cosine_similarity(u, v)
# print(cost)
print(np.rad2deg(np.arccos(cost[0][0])))
