import numpy as np

y_true = np.zeros((1, 10))
y_true[0,8] = 1
print(y_true)
x = np.array([1,3,4,5])
print(int(np.where(x==np.max(x))[0]))