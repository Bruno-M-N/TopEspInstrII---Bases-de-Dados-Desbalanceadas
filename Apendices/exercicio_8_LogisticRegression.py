from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe_lr_ix = make_pipeline (StandardScaler(), PCA(n_components = 2),
                         LogisticRegression(random_state = 1, solver='lbfgs'))
pipe_lr_ix.fit(X_train, y_train)
y_pred = pipe_lr_ix.predict(X_test)
print ('Acurácia do Teste: %.3f' %pipe_lr_ix.score(X_test, y_test))

print("")
printConfusionMatrix(y_true = y_test, y_pred = y_pred,
                     model = "Modelo LogisticRegression IX")
print("")
printClassificationReport(y_true = y_test, y_pred = y_pred,
                          model = "Modelo LogisticRegression IX")