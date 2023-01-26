import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.00001)
y = x ** 2
plt.plot(x, y)
plt.xlabel(' x variables ')
plt.ylabel('y  variables')
plt.show()
