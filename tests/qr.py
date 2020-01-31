import numpy as np

import math

def house(v):
    v = v.copy()
    sigma = np.dot(v[1:], v[1:])
    print('v = {}'.format(v))
    x0 = v[0]
    print('x0 = {}'.format(x0))
    if sigma > 0:
        mu = math.sqrt(x0*x0 + sigma)
        print('mu = {}'.format(mu))
        if x0 <= 0:
            v0 = x0 - mu
        else:
            v0 = -sigma/(x0 + mu)
        v02 = v0*v0
        print('v02 = {}'.format(v02))
        beta = 2*v02/(sigma + v02)
        print('beta = {}'.format(beta))
        v = v / v0
        v[0] = 1
        print('v = {}'.format(v))
    elif x0 < 0:
        beta = 2
        v[0] = 1
    else:
        beta = 0
        v[0] = 1
    return beta, v

A = np.array([[1, -1],[1, 1]])
n = 2

A0 = A
beta, v0 = house(A0[:,0])
print("v0 = {}".format(v0))

v0 = v0.reshape((n,1))
print("v0 = {}".format(v0))

rho = np.dot(v0.transpose(),A[:,1:])
print("rho = {}".format(rho))

Q0 = np.eye(n) - beta * np.dot(v0, v0.transpose())
print("Q0 = {}".format(Q0))

A1 = np.dot(Q0, A0)
print("A1 = {}".format(A1))

beta, v1 = house(A1[1:,1])
print("v1 = {}".format(v1))

v1 = np.vstack((0, v1))
print("v1 = {}".format(v1))

Q1 = np.eye(n) - beta * np.dot(v1, v1.transpose())
print("Q1 = {}".format(Q1))

A2 = np.dot(Q1, A1)
print("A2 = {}".format(A2))

Q = np.dot(Q0, Q1)
R = A2
R[1,0] = 0
print('Q = {}\nR = {}'.format(Q, R))

print(np.dot(Q, R) - A)

b = np.array([2, 1])
x1 = np.dot(Q.transpose(), b)
print('x1 = {}'.format(x1))
