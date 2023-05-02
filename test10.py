import numpy as np
import matplotlib.pyplot as plt

def PositionAtTemp(x, y, a, Tf):
    Tf = 1340
    mask0 = a <= Tf # lower node
    mask1 = a > Tf  # upper node
    a0 = a[mask0][-1]
    a1 = a[mask1][0]
    xy0 = np.array([x[mask0][-1], y[mask0][-1]])
    xy1 = np.array([x[mask1][0], y[mask1][0]])
    xy = ((a1 - 1341)*xy0 + (1340-a0)*(xy1))/(a1-a0)
    return xy

a = np.arange(1200.0, 1400.0, 20.0)
x = np.arange(0.0, 5.0, (5.0-0.0)/np.size(a))
y = np.arange(0.0, 2.0, (2.0-0.0)/np.size(a))

Tf = 1340
mask0 = a <= Tf
mask1 = a > Tf
a0 = a[mask0][-1]
xy0 = np.array([x[mask0][-1], y[mask0][-1]])
a1 = a[mask1][0]
xy1 = np.array([x[mask1][0], y[mask1][0]])
print(f'a0: {a0}, xy0:[{xy0}]')
print(f'a1: {a1}, xy1:[{xy1}]')
print(f'(a1 - 1341): {(a1 - 1341)}')
print(f'(a1 - 1341)*xy0: {(a1 - 1341)*xy0}')
xy = ((a1 - 1341)*xy0 + (1340-a0)*(xy1))/(a1-a0)
print(xy)

print(a)
print(x)
print(y)
print(mask1)
print(mask0)
print('a: {0}, x: {1}, y:{2}'.format(a[mask1][0], x[mask1][0], y[mask1][0]))
print('a: {0}, x: {1}, y:{2}'.format(a[mask0][-1], x[mask0][-1], y[mask0][-1]))
