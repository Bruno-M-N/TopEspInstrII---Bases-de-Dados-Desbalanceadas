import pandas as pd
# df = pd.read_csv('C:/Users/Usuário/Desktop/wdbc.data', header=None)
df = pd.read_csv(path + '/datasets/wdbc.data', header=None)
df.head()
df.shape
from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values
y =df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_
le.transform(['M', 'B'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,
                                                    stratify = y,
                                                    random_state = 1)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe_lr = make_pipeline (StandardScaler(), PCA(n_components=2),
                         LogisticRegression(random_state=1, solver='lbfgs'))
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print ('Acurácia do Teste: %.3f' %pipe_lr.score(X_test, y_test))

from sklearn.utils import resample
print('Número de Dados da Classe 0 (anterior):',
      X[y == 0].shape[0])
print('Número de Dados da Classe 1 (anterior):',
      X[y == 1].shape[0])
X_reamostrado, y_reamostrado = resample(X[y == 0],
                                        y[y == 0],
                                        replace=True,
                                        n_samples = X[y == 1].shape[0],
                                        random_state = 123)
print('Número de Dados da Classe 0 (após a reamostragem)',
      X_reamostrado.shape[0])

import numpy as np
x_bal = np.vstack((X[y == 1], X_reamostrado))
y_bal = np.hstack((y[y == 1], y_reamostrado))
print('x_bal.shape[0]',
      x_bal.shape[0])
y_pred = np.zeros(y_bal.shape[0])
np.mean(y_pred == y_bal) * 100

X_train, X_test, y_train, y_test = train_test_split(x_bal, y_bal,
                                                    test_size = 0.20,
                                                    stratify = y_bal,
                                                    random_state = 1)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe_lr = make_pipeline (StandardScaler(), PCA(n_components=2),
                         LogisticRegression(random_state=1, solver='lbfgs'))
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print ('Acurácia do Teste: %.3f' %pipe_lr.score(X_test, y_test))

print("")
printConfusionMatrix(y_true = y_test, y_pred = y_pred,
                     model = "Modelo LogisticRegression VII")
print("")
printClassificationReport(y_true = y_test, y_pred = y_pred,
                          model = "Modelo LogisticRegression VII")