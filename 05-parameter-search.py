from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

X, y = fetch_california_housing(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

param_distributions = {'n_estimators': randint(1, 5), 'max_depth': randint(5, 10) }
search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
        n_iter=5, 
        param_distributions=param_distributions,
        random_state=0)

search.fit(X_train, y_train)
print(search.best_params_)
