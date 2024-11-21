from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# y = a0 + a1x --> Linear Regression
# y = a0 + a1x1 + a2x2 + ...... + anxn --> multi variable linear regression 
# y = a0 + a1x1 + a2x2 

X = np.random.rand(100, 2)
coef = np.array([3,5])
# y = 0 + np.dot(X, coef)
y = np.random.rand(100) + np.dot(X, coef)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = "3d")
# # X in birinci sutunu - ikinci sutunu ve y olarak yaziyoruz.
# ax.scatter(X[:,0], X[:,1], y)
# ax.set_xlabel("X1")
# ax.set_ylabel("X2")
# ax.set_zlabel("y")

lin_reg = LinearRegression()
lin_reg.fit(X, y)


fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
# X in birinci sutunu - ikinci sutunu ve y olarak yaziyoruz.
ax.scatter(X[:,0], X[:,1], y)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")

X1, X2 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
y_pred = lin_reg.predict(np.array([X1.flatten(), X2.flatten()]).T)
ax.plot_surface(X1, X2, y_pred.reshape(X1.shape), alpha = 0.3)
plt.title("multi variable linear regression")

print("Kat Sayilar: ", lin_reg.coef_)
print("Kesisim: ", lin_reg.intercept_)












