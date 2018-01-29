
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
rng = np.random.RandomState(12)
#Creation de l'echantillon
X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=12, cluster_std=3)
X[:200], y[:200]

y[y==0] = -1
y[:200]

X_train = X[:1800]
y_train = y[:1800]
X_test = X[1800:]
y_test = y[1800:]

X_test[0], X_train[-1]


########
#Regression linéaire par rapport à l'échantillon crée
from sklearn.linear_model import LinearRegression

model_LR = LinearRegression(fit_intercept=True)

model_LR.fit(X_train, y_train)

np.all((model_LR.predict(X_train)>0) == (y_train>0))

def trace(X, y, w):
    plt.scatter(X[:,0], X[:,1], c=y, cmap='Paired', s=100)
    scale = [np.min(X[:,0]), np.max(X[:,0])]
    plt.plot(scale,[[(-w[0]-w[1]*i)/w[2]] for i in scale])

#trace(X_train, y_train, (model_LR.intercept_, *model_LR.coef_))

np.all((model_LR.predict(X_test)>0) == (y_test>0))
trace(X, y, (model_LR.intercept_, *model_LR.coef_))

np.sum((model_LR.predict(X_test)>0) == (y_test>0))/X_test.shape[0]

#Score de la regression effectuée
model_LR.score(X_test, y_test)

#voir graphiquement le résultat de la regression
#plt.show()


######Implementation du Perceptron

from sklearn.linear_model import Perceptron

model_P = Perceptron(random_state=12, max_iter=10000)

model_P.fit(X_train, y_train)

#Score du test
print(model_P.score(X_test, y_test))

trace(X, y, (*model_P.intercept_.flat, *model_P.coef_.flat))

plt.show()
