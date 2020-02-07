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