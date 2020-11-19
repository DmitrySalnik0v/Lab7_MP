import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('iris.csv')
X = data.values[:,0:4]
Y = data.values[:,4]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=100)
tree_entropy = DecisionTreeClassifier(criterion='entropy')
tree_entropy.fit(X_train,Y_train)
print('Точность (деревья) = ',tree_entropy.score(X_test,Y_test))
predicts = []
predicts.append(tree_entropy.predict([[3.2,2.5,2,0]]))
predicts.append(tree_entropy.predict([[1.2,4.5,2,1]]))
predicts.append(tree_entropy.predict([[1.2,3.2,4,6]]))
predicts.append(tree_entropy.predict([[4.2,1.2,4,6]]))
predicts.append(tree_entropy.predict([[3.2,1.2,1,2]]))
export_graphviz(tree_entropy,out_file='tree.dot')
##################################################
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,Y_train)
print('Точность (k ближайших соседей) = ',knn.score(X_test,Y_test))
predicts_knn = []
predicts_knn.append(knn.predict([[3.2,2.5,2,0]]))
predicts_knn.append(knn.predict([[1.2,4.5,2,1]]))
predicts_knn.append(knn.predict([[1.2,3.2,4,6]]))
predicts_knn.append(knn.predict([[4.2,1.2,4,6]]))
predicts_knn.append(knn.predict([[3.2,1.2,1,2]]))
print('Деревья - k ближайших соседей')
for i in range(len(predicts)):
    print(predicts[i],' - ',predicts_knn[i])