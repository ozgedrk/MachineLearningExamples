# sklearn : ML Library
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


# (1) Veri seti incelemesi

cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target

# (2) Makine Ogrenmesi Modelinin secilmesi-KNN Siniflandirmasi
# (3) Modelin Train Edilmesi

X = cancer.data   # features
y = cancer.target # target

knn = KNeighborsClassifier()  # Model olusturma komsu parametresini unutma *****
knn.fit(X , y) # fit fonskiyonu verimiz (samples + target) kullanarak KNN algoritmasini egitir

# (4) Sonuclarin Degerlendirilmesi

knn.predict(X)

# (5) Hiperparametre Ayarlamasi
