# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

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

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

import matplotlib.pyplot as plt
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()