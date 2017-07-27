import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def gender_to_int(x):
    if x == 'male':
        return 1
    return 0

data = pd.read_csv('data/train.csv').dropna()
data['Sex'] = data['Sex'].apply(gender_to_int)
data['Age'] = (data['Age'] - np.mean(data['Age'])) / np.std(data['Age'])
data['Fare'] = (data['Fare'] - np.mean(data['Fare'])) / np.std(data['Fare'])
test = pd.read_csv('data/test.csv').dropna()

vars = ['Age', 'Fare', 'Sex', 'Pclass', 'SibSp']

X_train, X_test, y_train, y_test = train_test_split(data[vars], data['Survived'], random_state=3)

reg = LinearRegression()

reg.fit(X_train, y_train)

out = [int(np.round(x)) for x in reg.predict(X_test)]

accuracy = []
for i in range(len(out)):
    accuracy.append(out[i] == y_test.values[i])

print(np.mean(accuracy))

