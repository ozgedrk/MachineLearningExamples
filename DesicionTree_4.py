from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np 
import matplotlib.pyplot as plt

# Create a data set

X = np.sort(5 * np.random.rand(80,1), axis = 0)
y = np.sin(X).ravel()
y[::5] += 0.5 * (0.5 - np.random.rand(16)) 


# plt.plot(X,y)

# plt.scatter(X,y)

regr_1 = DecisionTreeRegressor(max_depth = 2)
regr_1.fit(X, y)

regr_2 = DecisionTreeRegressor(max_depth = 15)
regr_2.fit(X, y)

X_test = np.arange(0, 5, 0.05)[:, np.newaxis]
y_pred_1 = regr_1.predict(X_test)
y_pred_2 = regr_2.predict(X_test)

plt.figure()
plt.plot(X, y, c = "red", label = "data")
plt.scatter(X, y, c = "red", label = "data")
plt.plot(X_test, y_pred_1, color = "blue", label = "Mac Depth: 2", linewidth = 2)
plt.plot(X_test, y_pred_2, color = "green", label = "Mac Depth: 15", linewidth = 2)

plt.xlabel("data")
plt.ylabel("target")
plt.legend()









