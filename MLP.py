import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
rng = np.random.RandomState(12)

from sklearn.neural_network import MLPClassifier
from sklearn.datasets.samples_generator import make_blobs, make_moons


from sklearn.model_selection import train_test_split


X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=12, cluster_std=3)
y[y==0] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=12)

model_MLP = MLPClassifier(hidden_layer_sizes=(4,), solver='sgd', learning_rate='constant', learning_rate_init=0.001 )

model_MLP.fit(X_train, y_train)

model_MLP.coefs_


model_MLP.predict(X_test) == y_test



def trace(X, y, model=None):
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    ax[0].scatter(X[:,0], X[:,1], c=y, cmap='Paired', s=100)
    if model:
        filter_errors = model.predict(X)!=y
        X_errors = X[filter_errors]
        ax[1].scatter(X[:,0], X[:,1], c=y, cmap='Paired', s=100)
        ax[1].scatter(X_errors[:,0], X_errors[:,1], marker="x", c='black', s=150)
        dxx, dyy = 0.01, 0.01
        xx = np.arange(1.1*np.min(X[:,0]), 1.1*np.max(X[:,0]), dxx)
        yy = np.arange(1.1*np.min(X[:,1]), 1.1*np.max(X[:,1]), dyy)
        XX, YY = np.meshgrid(xx, yy)
        Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.contourf(XX,YY,Z,cmap=plt.cm.coolwarm, alpha=0.2)


trace(X, y, model_MLP)
plt.show()
