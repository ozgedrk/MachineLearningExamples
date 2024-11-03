import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
x = np.linspace(-3,3,20)
y = 2*x + 3 + 3*np.random.randn(x.size)


x_test = np.linspace(-3,3,20)
y_test = 2.3*x_test + 3*np.random.randn(x_test.size) + 4
mdl = LinearRegression()

def fun(k):
    poly = PolynomialFeatures(degree=k)
    print(poly)
    x_poly = poly.fit_transform(x.reshape(-1,1))
    x_test_poly = poly.transform(x_test.reshape(-1,1))
    mdl.fit(x_poly,y)
    ypred = mdl.predict(x_test_poly)
    plt.plot(x_test,y_test, "*g")
    plt.plot(x,y,"*b")
    plt.plot(x_test, ypred, "r")
    plt.ylim([-3,11])
    plt.grid()
    print("Train: ", r2_score(y,mdl.predict(x_poly)))
    print("Test: ", r2_score(y_test,ypred))
fun(1)

# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score



df = pd.read_csv("Hitters_Data.csv")
df.info()
#df['Salary'].fillna(df['Salary'].mean(), inplace = True)
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
df.info()
df = pd.get_dummies(df, drop_first=True)
df

y = df["Hits"]
x = df.drop(["Hits"], axis=1)

# Regularization Overfittingi azaltir.

# Ridge Regression objective Function = ||y - Xw||^2_2 + alpha * ||w||^2_2


ridge_model = Ridge(alpha = 10000000000).fit(x, y) 
ridge_model.coef_



model_LR = LinearRegression()
model_LR.fit(x,y)
model_LR.coef_


# Split the data into training/testing sets
diabetes_X_train = pd.DataFrame([1,3])
diabetes_X_test = pd.DataFrame([1,2,3,4,5,6])

# Split the targets into training/testing sets
diabetes_y_train = pd.DataFrame([1,5])
diabetes_y_test = pd.DataFrame([2,2,3.5,8,5,9])

# Create linear regression object
regr = linear_model.LinearRegression()
regr_R = Ridge(alpha = 1)

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)
regr_R.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)
diabetes_y_pred_R = regr_R.predict(diabetes_X_test)

# The coefficients
print("Coefficients of linear regression: \n", regr.coef_)
print("Coefficients of linear Ridge regression: \n", regr_R.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Mean squared error of Ridge: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred_R))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))
print("Coefficient of determination of Ridge: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred_R))

# Plot outputs

plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=2)
plt.plot(diabetes_X_test, diabetes_y_pred_R, color="orange", linewidth=2)
plt.scatter(diabetes_X_train, diabetes_y_train, color="green")
plt.scatter(diabetes_X_test, diabetes_y_test, color="red")
plt.xticks(())
plt.yticks(())

plt.show()












