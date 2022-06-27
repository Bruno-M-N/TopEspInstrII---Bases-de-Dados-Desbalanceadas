from sklearn.model_selection import GridSearchCV # RandomizedSearchCV

param_C_range = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_gamma_range = [0.0001, 0.001, 0.01, 0.1, 1.0]
param_grid = [
              # {'svc__C': param_C_range, 'svc__kernel': ['linear']},
              {'svc__C': param_C_range, 'svc__gamma': param_gamma_range,
               'svc__kernel': ['rbf']}]
gs = GridSearchCV(estimator = pipe_svc, param_grid = param_grid,
                  scoring = 'accuracy', refit = True, cv = 5, n_jobs = -1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
clf = gs.best_estimator_
print('Accuracy do Teste: %.3f' %clf.score(X_test, y_test))