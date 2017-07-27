import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

def gender_to_int(x):
    if x == 'male':
        return 10
    return 0

data = pd.read_csv('data/train.csv').dropna()
data['Sex'] = data['Sex'].apply(gender_to_int)
data['Age'] = (data['Age'] - np.mean(data['Age'])) / np.std(data['Age'])
data['Fare'] = (data['Fare'] - np.mean(data['Fare'])) / np.std(data['Fare'])
test = pd.read_csv('data/test.csv').dropna()

vars = ['Age', 'Fare', 'Sex', 'Pclass']

X_train, X_test, y_train, y_test = train_test_split(data[vars], data['Survived'], random_state=3)

knn = KNeighborsClassifier(13)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(knn.score(X_test, y_test))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Load model
joblib.dump(knn, 'model.pkl', compress=9)
knn = joblib.load('model.pkl')

knn.predict(X_test)