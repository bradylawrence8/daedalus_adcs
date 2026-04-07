import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import FormatStrFormatter

import time

start_time = time.perf_counter()

def rotx(theta):
    return np.array([[1, 0, 0], [0, math.cos(theta), math.sin(theta)], [0, -math.sin(theta), math.cos(theta)]])

def roty(theta):
    return np.array([[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0], [math.sin(theta), 0, math.cos(theta)]])

def rotz(theta):
    return np.array([[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

def plotAxis(m, ax, style):
    ax.plot([0, m[0, 0]], [0, m[1, 0]], [0, m[2, 0]], color='red', linestyle=style)
    ax.plot([0, m[0, 1]], [0, m[1, 1]], [0, m[2, 1]], color='blue', linestyle=style)
    ax.plot([0, m[0, 2]], [0, m[1, 2]], [0, m[2, 2]], color='green', linestyle=style)

def plotVec(v, ax, c, style):
    ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=c, linestyle=style)

def sunvec(c, sz, dcm):
    v = np.array([c[0]-sz[0]/2, sz[1]/2-c[1], 0])
    vn = v/np.linalg.norm(v)
    n = np.array([vn[1], -vn[0], vn[2]])
    return np.matmul(dcm, n)

def powMeth(A,q): #matrix A and initial guess for normalized eigenvector q, returns positive eivenvalue
    #creating A^2
    A2 = np.matmul(A,A)

    #power method
    lambda_0 = 0
    lambda_k = 1e6
    q_k = q
    while np.abs(lambda_k - lambda_0) >= 1e-8:
        z = np.matmul(A2,q_k)
        q_k = z / np.linalg.norm(z,2)
        lambda_0 = lambda_k
        lambda_k = np.matmul(np.transpose(np.conjugate(q_k)),np.matmul(A2,q_k))
    lambda_k = np.sqrt(lambda_k)

    #finding eigenvectors
    u1 = np.matmul(A,q_k) + lambda_k * q_k
    u2 = np.matmul(A,q_k) - lambda_k * q_k
    u1norm = u1 / np.linalg.norm(u1,2)
    u2norm = u2 / np.linalg.norm(u2,2)

    return u1norm

def qmethod(weights, vb, vf):
    B = np.zeros((3, 3))
    for i, w in np.ndenumerate(weights):
        B = B + np.multiply(np.matmul(np.transpose(vb[i, :]), vf[i, :]), w)
    S = B + np.transpose(B)
    s = np.trace(B)
    z = np.array([[B[1, 2]-B[2, 1]], [B[2, 0]-B[0, 2]], [B[0, 1]-B[1, 0]]])
    Ss = S-np.multiply(s, np.identity(3))
    K = np.concatenate([np.concatenate([Ss, z], 1), np.transpose(np.array([z[0], z[1], z[2], np.array([s])]))])
    return K

dcm1 = np.matmul(rotx(-math.pi/2), roty(0))
dcm2 = np.matmul(rotx(-math.pi/2), roty(math.pi/2))

ax = plt.figure().add_subplot(projection='3d')

plotAxis(np.identity(3), ax, 'solid')
plotAxis(dcm2, ax, 'dotted')
plt.axis([-2, 2, -2, 2, -2, 2])

size1 = np.array([4032, 3024])
#size2 = np.array([800, 600])
c1 = np.array([1000, 1000])
c2 = np.array([50, 2800])

sv1 = sunvec(c1, size1, dcm1)
sv2 = sunvec(c2, size1, dcm2)
S = np.cross(sv1, sv2)/np.linalg.norm(np.cross(sv1, sv2))


    
plotVec(sv1, ax, 'yellow', 'dashed')
plotVec(sv2, ax, 'cyan', 'dashed')
plotVec(S, ax, 'magenta', 'solid')
plotVec(-S, ax, 'magenta', 'solid')
print(S)



#true = np.array([0.40619892,0.8609375,0.30625])
#ax.plot(true[0], true[1], true[2], 'ko')
#dcm3 = np.matmul(rotx(math.pi/2), roty(math.pi/2), rotz(math.pi/4))
#K = qmethod(np.array([0.5, 0.5]), np.array([S, [0, 0, -1]]), np.array([np.matmul(dcm3, S), np.matmul(dcm3, np.array([0, 0, -1]))]))


end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"Execution time: {elapsed_time:.4f} seconds")


plt.show()