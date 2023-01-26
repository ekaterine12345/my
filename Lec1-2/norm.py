import numpy as np
from numpy.linalg import norm
from numpy import inf

u = np.array([1, 2, 3])
norm2 = norm(u)
norm1 = norm(u, 1)
print(norm2, type(norm2))  # ნორმა 2   ვექტროის სიგრძე
print(norm1, type(norm1))  # ნორმა 1   ვექტროის განზომილელების აბსოლიტური მნიშვნელობების ჯამი

a = np.array([1, 3, 0, 2, 0, 2, 32, 3, 23, 0, 10, 110, 100,  555])
print(np.count_nonzero(a))  # ნორმა 0


v = np.array([-1, -2, -3, 100, 10])  # უსასრულო რიგის ნორმა. მაქსიმალური აბსოლიტური მნიშვნელობა
norm1 = norm(v, inf)
print(norm1)
print(np.abs(v))
print(np.max(np.abs(v)))  # ბოლო რიგის ნორმა





