import numpy as np
import matplotlib

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# arr_1 = np.array((2, 33, 43, 1111))
#                 0   1  2  3   4   5  6     7   8    9     10
ar2 = np.array([100, 12, 3, 23, 1, 90, 91, 100, 11, 100, 10000])
# a = [2, 3, 4]

# print((arr_1, type(arr_1)))
# print(a, type(a))


print(np.__version__)
print(ar2, type(ar2))
print(ar2, ar2.dtype)
print(ar2[2:5])
print(ar2[2:9:2])
print(ar2[0], ar2[10], ar2[0]+ar2[1])
for each in ar2:
    print(each, end="; ")
