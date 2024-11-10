import numpy as np

a = np.array([1,2,3,4,5])
b = np.array([5,4,3,2,1])

c = [a, b]

c[0][0] = 0

print(a)

c = np.empty(2, dtype=object)
c[:] = [a,b]

c[0][0] = 1

print(a)
