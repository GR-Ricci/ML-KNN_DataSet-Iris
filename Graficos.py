from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


iris = datasets.load_iris()

features_all = []
sepal_len = []
sepal_width = []
petal_len =[]
petal_width = []

features = iris.data
targets = iris.target
group = ('Iris_Setosa', 'Iris_versicolor', 'Iris_virginica')
colors = ('blue','green','red')

for v in features:
    sepal_len.append(v[0])
    sepal_width.append(v[1])

for v in features:
    petal_len.append(v[2])
    petal_width.append(v[3])

data = ((sepal_len[:50], sepal_width[0:50]), (sepal_len[50:100], sepal_width[50:100]),
        (sepal_len[100:150], sepal_width[100:150]))
data_petal = ((petal_len[:50], petal_width[0:50]), (petal_len[50:100], petal_width[50:100]),
        (petal_len[100:150], petal_width[100:150]))

plt.figure()
for v, c, g in zip(data,colors,group):
    x0,y0 = v #criando variavel para receber
    plt.scatter(x0,y0, color = c, alpha=1, label = g)
    plt.title('Iris Dataset Scatter Plot')
plt.title('Gráfico Sepal')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()

plt.figure()
for v, c, g in zip(data_petal,colors,group):
    xp0,yp0 = v
    plt.scatter(xp0,yp0, color = c, alpha=1, label= g)
    plt.title('Iris Dataset Scatter Plot')
plt.title('Gráfico Petal')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()



#dataframe
df_iris = pd.DataFrame(np.column_stack((iris.data, iris.target)), columns = iris.feature_names+['target'])
pd.set_option('display.max_columns', None)  # mostra todas as colunas sem cortar
pd.set_option('display.width', 1000)

print(df_iris)
print('----------')
print(df_iris.describe())

plt.show()