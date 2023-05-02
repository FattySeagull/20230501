# cook your dish here
import numpy as np

def inter(a, v):
    l = [(up-v)*(v-low)>0.0 for up, low in zip(a[1::], a)]
    l = np.append(l, False)
    return l
    
a = np.arange(1,10,1)
b = a[1:]
c = b - a[0::-1]
d = inter(a, 5.5)
e = np.roll(d, 1)
print(a)
print(b)
print(c)
print(d)
print(e)

print(a[d][0])
