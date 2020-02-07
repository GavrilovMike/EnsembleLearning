from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
X, y = load_iris(return_X_y=True)
print('X: ', X, '\n')
print('y: ', y, '\n')
print('X_shape: ', X.shape, '\n')
print('y_shape: ', y.shape, '\n')
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svr', make_pipeline(StandardScaler(),
                          LinearSVC(random_state=42)))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)
print('Test: \n', clf.fit(X_train, y_train).score(X_test, y_test), '\n')

print('X_train: ', X_train, '\n')
print('X_test: ', X_test, '\n')
print('y_train: ', y_train, '\n')
print('y_test: ', y_test, '\n')