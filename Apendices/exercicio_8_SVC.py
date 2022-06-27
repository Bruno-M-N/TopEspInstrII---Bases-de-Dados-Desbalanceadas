from sklearn.svm import SVC

# criação de pipeline
pipe_svc = make_pipeline(StandardScaler(), PCA(n_components = 2),
                         SVC(C = 1, kernel = 'rbf', gamma = 0.001,
                             cache_size = 500, random_state = 1))
# Treinamento
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)

print ('Acurácia do Teste: %.3f' %pipe_svc.score(X_test, y_test))

print("")
printConfusionMatrix(y_true = y_test, y_pred = y_pred,
                     model = "Questão IX Modelo SVC")
print("")
printClassificationReport(y_true = y_test, y_pred = y_pred,
                          model = "Questão IX Modelo SVC")