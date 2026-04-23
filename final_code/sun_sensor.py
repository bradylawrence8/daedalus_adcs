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