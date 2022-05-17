from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
import time


def fit_SGDClassifier(X_train_tfidf, y_train, class_weight=None):
    start_time = time.time()
    grid_params = {"loss": ["hinge", "log", "modified_huber"],
                   "penalty": ["l1", "l2", "elasticnet"],
                   "alpha": [1e-4, 1e-5],
                   }
    grid = GridSearchCV(SGDClassifier(class_weight=class_weight), grid_params, refit=True, cv=3, verbose=1)
    grid.fit(X_train_tfidf, y_train)
    print("Model take {} S".format(time.time()-start_time))
    return grid.best_estimator_, grid.best_params_