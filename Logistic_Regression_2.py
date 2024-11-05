import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve


df = pd.read_table('diabetesdata.txt')

X = df.drop('Diabetes', axis =1)
y = df['Diabetes']


logmodel = LogisticRegression(max_iter = 100, penalty = None)
logmodel.fit(X, y)

# beta (beta1 beta2 ve beta3) degerlerini gormek icin yaziyoruz
logmodel.coef_

# beta0 icin
logmodel.intercept_

# En cok etkileyen veriDPF
Model_intercept = pd.DataFrame({"Variables":'Intercept',"Coefficients":logmodel.intercept_[0]},index=[0])
Model_coefficients = pd.DataFrame({"Variables":X.columns,"Coefficients":np.transpose(logmodel.coef_[0])})
Model_coefficients = pd.concat([Model_intercept,Model_coefficients]).reset_index(drop=True)
print(Model_coefficients)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


X_test1 = pd.DataFrame([{'Preg': 1, 'Glucose': 100,'BP':25 ,'SkinThick':70,'Insul':50, 'BMI': 30, 'DPF': 0.2, 'Age':25 }])
X_test1

logmodel = LogisticRegression(max_iter=1000,penalty = None)
logmodel.fit(X_train,y_train)

y_pred1 = logmodel.predict(X_test1)
y_pred1


np.array(y_test)


y_pred = logmodel.predict(X_test)
score = logmodel.score(X_test, y_test)
print(score)


Model_coefficients = pd.DataFrame({"Variables":X.columns,"Coefficients":np.transpose(logmodel.coef_[0])})
Model_coefficients = pd.concat([Model_intercept,Model_coefficients]).reset_index(drop=True)
print(Model_coefficients)


cnf_matrix = confusion_matrix(y_test, y_pred)
cnf_matrix


def plot_conf_mat(y_test, y_preds):
    """
    Plots a nice looking confusion matrix using Seaborn's heatmap()
    """
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True,
                     cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

plot_conf_mat(y_test, y_pred)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


logmodel.predict_proba(X_test)


y_pred_proba = logmodel.predict_proba(X_test)[::,1]
y_pred_proba


threshold = 0.1
y_pred = (y_pred_proba > threshold).astype('float')
plot_conf_mat(y_test, y_pred)



fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

df = pd.read_table('diabetesdata.txt')

# %%


df = pd.read_table('diabetesdata.txt')

X=df[['Glucose','BMI']]
y = df['Diabetes']


X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1,stratify=y)

classifier=LogisticRegression(penalty= None)
classifier.fit(X_train,y_train)
print(classifier.coef_)
classifier.intercept_


def plot_dec_boundary(estimator,X,Y,h):
    X=np.array(X)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    print(Z)
    plt.figure(1, figsize=(10, 10))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    xcor = np.linspace(0,200,200)
    ycor = -xcor*0.46+98.7
    plt.plot(xcor, ycor)
    plt.show()
    

    
plot_dec_boundary(classifier,X_train,y_train,h=0.25)    



from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train,y_train)
plot_dec_boundary(classifier,X_train,y_train,h=0.25)




