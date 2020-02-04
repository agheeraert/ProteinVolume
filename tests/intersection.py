from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)
points = np.random.random((4, 3))*4-2
faces = [points[:4], points[1:], points[[0,1,3]], points[[0,2,3]]]
rvdw = np.random.random(1)

u = np.linspace(0, np.pi, 30)
v = np.linspace(0, 2 * np.pi, 30)
x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(Poly3DCollection(faces, color='g', alpha=0.5))
ax.plot_wireframe(x,y,z)
plt.show()