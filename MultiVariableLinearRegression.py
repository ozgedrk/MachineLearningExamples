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

# %%
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

#    squared : bool, default=True
#    If True returns MSE value, if False returns RMSE value.

rmse = mean_squared_error(y_test, y_pred, squared = False)
print("rmse: ", rmse)


# Gerçek değerler vs Tahmin edilen değerler
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Gerçek vs Tahmin")
plt.show()





































