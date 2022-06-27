# criação de pipeline
pipe_random_forest = make_pipeline(StandardScaler(), PCA(n_components = 2),
                                   RandomForestClassifier(random_state = 1))
# Treinamento
pipe_random_forest.fit(X_train, y_train)
y_pred = pipe_random_forest.predict(X_test)

print ('Acurácia do Teste: %.3f' %pipe_random_forest.score(X_test, y_test))
print ('Random Forest \nAccuracy:{0:.3f}'.format(accuracy_score(y_test,
                                                                y_pred)))

print("")
printConfusionMatrix(y_true = y_test, y_pred = y_pred,
                     model = "Questão III Modelo Random Forest")
print("")
printClassificationReport(y_true = y_test, y_pred = y_pred,
                          model = "Questão III Modelo Random Forest")