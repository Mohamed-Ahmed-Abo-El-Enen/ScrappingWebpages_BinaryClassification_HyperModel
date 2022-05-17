from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time


def fit_SVMClassifier(X_train_tfidf, y_train, class_weight=None):
    start_time = time.time()
    grid_params = {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                   "degree": [1, 2, 3],
                   "C": [1, 0.8, 0.5]}
    grid = GridSearchCV(SVC(class_weight=class_weight), grid_params, refit=True, cv=3, verbose=1)
    grid.fit(X_train_tfidf, y_train)
    print("Model take {} S".format(time.time()-start_time))
    return grid.best_estimator_, grid.best_params_