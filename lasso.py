import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
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

lasso = Lasso(alpha=.02, normalize=True)

lasso.fit(X_train, y_train)

lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(vars)), lasso_coef)
plt.xticks(range(len(vars)), vars, rotation=60)
plt.margins(0.02)
plt.show()

