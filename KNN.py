from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


iris_base = datasets.load_iris()
iris = pd.DataFrame(iris_base.data, columns = iris_base.feature_names)
iris['class'] = iris_base.target

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(iris.head())
print('-'*100)

#----- Pré processamentos ------
x = iris.iloc[:, :-1].values
y = iris.iloc[:,4].values

#----- Divisao para teste e treino ----
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20)

#----- scaliing e normalizacao -----
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

error = []

for i in range (1,40):
    knn = KNeighborsClassifier (n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12,6))
plt.plot(range(1,40), error, color= 'red',
         linestyle = 'dashed', marker ='o', markerfacecolor='blue',markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


#--Avaliação com base no melhor K
melhor_k = error.index(min(error)) + 1
print(f"Melhor valor de K: {melhor_k}")

# Treina o modelo com o melhor K
knn = KNeighborsClassifier(n_neighbors=melhor_k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

# Avaliação
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Lista dos melhores
min_error = min(error)
ks_bons = [i+1 for i, val in enumerate(error) if val == min_error]
print(f"K(s) com menor erro: {ks_bons}")

plt.show()