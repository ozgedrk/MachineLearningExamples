import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
#%matplotlib inline


df = pd.read_csv("Ev_Fiyatları.txt")

df.head()
df.info()
df.describe()

sns.pairplot(df)
plt.show()

sns.histplot(df['Ev Fiyatı'])


# corr yani corelasyon iki farkli degiskenin birbirlerine olan etkisi
#   sns.heatmap(df.corr(), annot=True)   (numeric hatasi veriyor)

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[float, int])

# Plot the heatmap
sns.heatmap(numeric_df.corr(), annot=True)
plt.show()


X = df.drop(["Ev Fiyatı","Adres"], axis = 1)
y = df['Ev Fiyatı']
X.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)


model = LinearRegression()
model.fit(X_train, y_train)

# Kesisim yani Y ekseni ile Lineer Regresyon dogrusunun kesistigi yer
print(model.intercept_)


Katsayılar = pd.DataFrame(model.coef_,X.columns,columns=['Katsayılar (Beta)'])
Katsayılar


tahminler = model.predict(X_test)
plt.scatter(y_test,tahminler)


sns.distplot((y_test-tahminler),bins=50)

sns.histplot((y_test-tahminler),bins=50)

lm = sm.OLS(y,X)
model = lm.fit()
model.summary()











