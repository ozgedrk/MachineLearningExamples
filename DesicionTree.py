from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score , confusion_matrix

import matplotlib.pyplot as plt


# Veri Seti Inceleme Ve Analiz
iris = load_iris()

#

X = iris.data    # features
y = iris.target  # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)

# Desicion Tree (Karar Agaci) modeli olustur ve train et

tree_clf = DecisionTreeClassifier(criterion="gini", max_depth= 5, random_state = 42) # criterion="entropy"
tree_clf.fit(X_train, y_train)


# Desicion Tree (Karar Agaci) evulation test
y_predict = tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print("Iris veri seti ile egitilen DT modeli dogrulugu:", accuracy)

conf_matrix = confusion_matrix(y_test, y_predict)
print("conf_matrix:")
print(conf_matrix)

plt.figure(figsize=(15,10)) 
plot_tree(tree_clf, filled= True , feature_names= iris.feature_names , class_names= list(iris.target_names))
plt.show()


feature_importances = tree_clf.feature_importances_

feature_names = iris.feature_names

# Feature degerlerini onem ve etkisine gore siralama

feature_importances_sorted = sorted(zip(feature_importances , feature_names), reverse=True)
for importance, feature_name in feature_importances_sorted:
    print(f"{feature_name} : {importance}")























