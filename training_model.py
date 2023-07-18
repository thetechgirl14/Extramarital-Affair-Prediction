from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf_LR = LogisticRegression(max_iter=1000)
    clf_LR.fit(X_train, y_train)
    return clf_LR, X_train, y_train, X_test, y_test

