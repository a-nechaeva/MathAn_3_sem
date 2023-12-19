import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # параметры контейнера для вывода графика

# Подготовка данных
X = np.arange(-1, 3, 0.1)
Y = np.array(X)
X, Y = np.meshgrid(X, Y)    # расширение векторов X,Y в матрицы
Z = X**3 * Y**2 * (4 - X - Y)

# Построение графика
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')    # метод для отрисовки графиков с параметрами по умолчанию

ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
# ax.set_zlim(-100, 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
