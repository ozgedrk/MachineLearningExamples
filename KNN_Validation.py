import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

df = pd.read_table("diabetesdata.txt")


X = np.array(df.drop(columns=['Diabetes']))

# Seperate target value
y = np.array(df['Diabetes']) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5, stratify = y )


# CV yani Cross Validation
cv = KFold(n_splits = 5, random_state = 1, shuffle = True)
CVAccuracy = []
for train_index, validation_index in cv.split(X_train):
    X_train_v, X_valid, y_train_v, y_valid = X[train_index], X[validation_index], y[train_index], y[validation_index]
    knn = KNeighborsClassifier(n_neighbors = 12)
    knn.fit(X_train_v, y_train_v)
    CVAccuracy.append([knn.score(X_valid, y_valid)])
df = pd.DataFrame(CVAccuracy, columns =['Validation Accuracy'])

FivefoldCVError = df.mean()

CVAccuracy = []

for j in range(1,26):
    for train_index, validation_index in cv.split(X_train):
        X_train_v, X_valid, y_train_v, y_valid = X[train_index], X[validation_index], y[train_index], y[validation_index]
        knn = KNeighborsClassifier(n_neighbors = 3)
        #  Fit the classifier to the data
        knn.fit(X_train_v, y_train_v)
        CVAccuracy.append([knn.score(X_valid, y_valid),j])
    df = pd.DataFrame(CVAccuracy, columns = ['Validation Accuracy','NeighbourSize'])
pd.set_option('display.max_rows',None)

kfoldCV = df.groupby("NeighbourSize")
kfoldCV = kfoldCV.mean()
kfoldCV = kfoldCV.reset_index()
kfoldCV[['NeighbourSize', 'Validation Accuracy']]
kfoldCV.loc[kfoldCV['Validation Accuracy'].idxmax()]




fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(kfoldCV['NeighbourSize'].values,kfoldCV['Validation Accuracy'].values,label = '5 fold CV Accuracy')
ax.set_xlabel('Komşu Sayısı')
ax.set_ylabel('Doğruluk')
ax.tick_params(axis='x', labelsize=8)
ax.legend(loc='best')


X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

knn = KNeighborsClassifier(n_neighbors=12)
scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())

scores

scores = cross_val_score(knn, X_train, y_train, cv=LeaveOneOut(), scoring='accuracy')

scores.mean()


LOOCVAccuracy=[]
for j in range(1,26):
    knn = KNeighborsClassifier(n_neighbors = j)
    scores = cross_val_score(knn, X_train, y_train, cv=LeaveOneOut(), scoring='accuracy')
    LOOCVAccuracy.append([scores.mean(),j])
LOOCV = pd.DataFrame(LOOCVAccuracy,columns=['Validation Accuracy','NeighbourSize'])
print(LOOCV)


fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(LOOCV['NeighbourSize'].values,LOOCV['Validation Accuracy'].values,label = 'Leave One Out Error')
ax.set_xlabel('Komşu Sayısı')
ax.set_ylabel('Doğruluk')
ax.tick_params(axis='x', labelsize=8)
ax.legend(loc='best')


knn = KNeighborsClassifier(n_neighbors=18)
knn.fit(X_train,y_train)
knn.score(x_test,y_test)



















