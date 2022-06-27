from sklearn.svm import SVC

# criação de pipeline
pipe_svc_otimizada = make_pipeline(StandardScaler(), PCA(n_components = 2),
                         SVC(C = 100, kernel = 'rbf', gamma = 1.0,
                             cache_size = 500, random_state = 1))
# Treinamento
pipe_svc_otimizada.fit(X_train, y_train)
y_pred = pipe_svc_otimizada.predict(X_test)

print ('Acurácia do Teste: %.3f' %pipe_svc_otimizada.score(X_test, y_test))

print("")
printConfusionMatrix(y_true = y_test, y_pred = y_pred,
                     model = "Questão IX Modelo SVC Otimizado")
print("")
printClassificationReport(y_true = y_test, y_pred = y_pred,
                          model = "Questão IX Modelo SVC Otimizado")