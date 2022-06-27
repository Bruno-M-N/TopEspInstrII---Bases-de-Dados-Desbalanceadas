import pandas as pd
# df = pd.read_csv('C:/Users/Usuário/Desktop/wdbc.data', header=None)
# df = pd.read_csv(path + '/datasets/wdbc.data', header=None)
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
pipe_lr = make_pipeline (StandardScaler(), PCA(n_components = 2),
                         LogisticRegression(random_state = 1, solver='lbfgs'))
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print ('Acurácia do Teste: %.3f' %pipe_lr.score(X_test, y_test))

print("")
printConfusionMatrix(y_true = y_test, y_pred = y_pred,
                     model = "Modelo LogisticRegression")
print("")
printClassificationReport(y_true = y_test, y_pred = y_pred,
                          model = "Modelo LogisticRegression")