import numpy as np
A = np.array([[1, 0, 1], [1, 2, 2], [1, 0, 2]])
rank = np.linalg.matrix_rank(A)
print(rank)