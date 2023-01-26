from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
grid_search.fit(X_train, y_train)
print(grid_search.score(X_test, y_test))
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)  # საუკეთესო მოდელი
# print(grid_search.cv_results_) # გვიბრუნებს ძებნის ყველა ასპექტს, გამოთვლილ ქულებს და ასე შემდეგ