import numpy as np
a = np.array([[2, -3], [1, 5]])
b = np.array([[4], [7]])
ainv = np.linalg.inv(a)
solution = np.dot(ainv, b)
print(solution)
print(ainv)

# მეორე გზა
A = np.array([[2, -3], [1, 5]])
b = np.array([4, 7])
solution_2 = np.linalg.solve(A, b)
print(solution_2)


