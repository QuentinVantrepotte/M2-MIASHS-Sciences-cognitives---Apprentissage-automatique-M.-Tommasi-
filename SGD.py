
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
rng = np.random.RandomState(12)
#Creation de l'echantillon
X, y = make_blobs(n_samples=2000, centers=2, n_features=2, random_state=12, cluster_std=3)
X[:200], y[:200]

y[y==0] = -1
y[:200]

X_train = X[:1800]
y_train = y[:1800]
X_test = X[1800:]
y_test = y[1800:]

X_test[0], X_train[-1]

from sklearn.linear_model import SGDClassifier

#
model_SGD = SGDClassifier(random_state=12, max_iter=100000)

model_SGD.fit(X_train, y_train)

print(model_SGD.score(X_train, y_train))
print(model_SGD.score(X_test, y_test))

def trace(X, y, w):
    plt.scatter(X[:,0], X[:,1], c=y, cmap='Paired', s=100)
    scale = [np.min(X[:,0]), np.max(X[:,0])]
    plt.plot(scale,[[(-w[0]-w[1]*i)/w[2]] for i in scale])

trace(X, y, (*model_SGD.intercept_.flat, *model_SGD.coef_.flat))


#On change de fonction : ici on fait avec une squared loss
model_SGD = SGDClassifier(loss="squared_loss", max_iter=100000,
                          random_state=12)

model_SGD

model_SGD.fit(X_train, y_train)
print(model_SGD.score(X_train, y_train))
print(model_SGD.score(X_test, y_test))

trace(X_train, y_train, (*model_SGD.intercept_.flat, *model_SGD.coef_.flat))

#Ici on fait un pas de 0.1 pour le SGD => mauvais résultat
model_SGD = SGDClassifier(loss="squared_loss", max_iter=100000,
                learning_rate='constant', eta0=0.1, random_state=12)
model_SGD.fit(X_train, y_train)
print(model_SGD.score(X_train,y_train))
print(model_SGD.score(X_test, y_test))
trace(X, y, (*model_SGD.intercept_.flat, *model_SGD.coef_.flat))

#voir graphiquement le résultat de la regression
plt.show()
