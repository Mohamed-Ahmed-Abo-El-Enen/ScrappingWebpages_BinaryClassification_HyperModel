from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import time


def fit_XGBClassifier(X_train_tfidf, y_train, class_weight=None):
    start_time = time.time()
    grid_params = {'learning_rate': [0.001, 0.01, 0.05],
                   'max_depth': [5, 10, 15],
                   'n_estimators': [5, 10, 15]
                   }
    grid = GridSearchCV(XGBClassifier(sample_weight=class_weight), grid_params, refit=True, cv=3, verbose=1)
    grid.fit(X_train_tfidf, y_train)
    print("Model take {} S".format(time.time()-start_time))
    return grid.best_estimator_, grid.best_params_