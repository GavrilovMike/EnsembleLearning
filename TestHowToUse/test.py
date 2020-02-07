from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
import sklearn

# X, y = load_iris(return_X_y=True)
# clf = AdaBoostClassifier(n_estimators=100)
# scores = cross_val_score(clf, X, y, cv=5)
# print('X = ', X, '\n')
# print('Y = ', y, '\n')
# print(scores.mean())
# fitted = clf.fit(X,y)
# print('Fitted ', fitted, '\n')
# stageg_score = clf.score(X,y);
# print('Staged Scoere: ', stageg_score, '\n')
# decision_function = clf.decision_function(X)
# print('Decision Function ', decision_function, '\n')
# stageg_decision_function = clf.staged_decision_function(X)
# print('Staged Decision Function ', stageg_decision_function, '\n')
# stageg_predict = clf.staged_predict(X)
# print('Staged Predict ', stageg_predict, '\n')
# predict = fitted.predict(X)
# print('Predict ', predict, '\n')
# print('Check', predict == y, '\n')
# predict_proba = clf.predict_proba(X)
# print('Predict Proba ', predict_proba, '\n')

from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print ('X_train ', X_train , '\n')
print ('X_test ', X_test , '\n')
print ('y_train ', y_train , '\n')
print ('y_test ', y_test , '\n')

abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)

model = abc.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print ('y_pred ', X_train , '\n')

# print ('Check ', y_pred == X_test, '\n')