import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from sklearn.datasets import load_iris

iris_data = load_iris()
iris_data
iris_data.keys()

iris_data['data']
iris_data['target']
iris_data['target_names']

print(iris_data['DESCR'])

iris_data['feature_names']

iris_df = pd.DataFrame(iris_data['data'], columns = iris_data['feature_names'])
iris_df

print(iris_df)

# display(iris_df)

X=iris_df
y=iris_data['target']

iris_data['target'].shape

iris_df.shape

X_train = X.iloc[:105, :]
y_train = y[:105]
X_test = X.iloc[105:, :]
y_test = y[105:]

y_test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

pd.plotting.scatter_matrix(iris_df, c=y, figsize=(15,15), marker='0',alpha=1)

model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X_train, y_train)
pre=model.predict(X_test)
pre

metrics.accuracy_score(pre, y_test)
model.score(X_test, y_test)

test_list = []
train_list = []

for k in range(1, 80, 2) :
    model = KNeighborsClassifier(n_neighbors= k)
    model.fit(X_train, y_train)
    test.append(model.score(X_test, y_test))

    train.append(model.score(X_train, y_train))

test_list
train_list

plt.figure(figsize= (7,7))
plt.plot(range(1, 80, 2), train_list, label = 'train')
plt.plot(range(1, 80, 2), test_list, label = 'test')
plt.legend()
st.pyplot()