import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Veri oluşturma
X = 4 * np.random.rand(100, 1)  # X 2D bir array
y = 2 + 3 * X**2 + np.random.randn(100, 1)  # Biraz gürültü ekledik

# PolynomialFeatures ile X'i genişletme
poly_feat = PolynomialFeatures(degree=2)  # 2. derece polinom
X_poly = poly_feat.fit_transform(X)

# Lineer regresyon modeli
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)  # Burada X_poly kullanılır

# Modeli çizmek için test verisi oluşturma
X_test = np.linspace(0, 4, 100).reshape(-1, 1)
X_test_poly = poly_feat.transform(X_test)
y_pred = poly_reg.predict(X_test_poly)

# Veri görselleştirme
plt.scatter(X, y, color="blue", label="Veriler")
plt.plot(X_test, y_pred, color="red", label="Polinom Regresyon")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polinom Regresyon Modeli")
plt.legend()
plt.show()
