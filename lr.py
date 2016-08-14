import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt




def log1pexp(x):
    seed = 33.3
    y = np.copy(x)
    idx = x < seed
    y[idx] = np.log1p(np.exp(x[idx]))
    return y


# def sigmoid(x):
#     return np.exp(-np.logaddexp(0, -x))

def sigmoid(x):
    return np.exp(-log1pexp(-x))


def lr_grad(X, y, w, gamma):
    a = X @ w
    z = sigmoid(a)
    g = (z - y) @ X + gamma * w
    return g


def lr_llh(X, y, w, gamma):
    n = X.shape[0]
    h = 2 * y - 1
    a = X @ w
    llh = -(np.sum(log1pexp(-h * a)) + 0.5 * gamma * np.dot(w, w)) / n
    return llh


def lr_gd(X, y, gamma=0.1, alpha=0.1):
    n = X.shape[0]
    x0 = np.ones((n, 1))
    X = np.hstack((x0, X))
    d = X.shape[1]
    epoch = 200
    llh = np.full(epoch, np.inf)
    w = np.zeros(d)
    for t in range(epoch):
        g = lr_grad(X, y, w, gamma)
        w = w - alpha*g
        llh[t] = lr_llh(X, y, w, gamma)
    return (w, llh)

def sampling(X, y, k):
    n = X.shape[0]
    idx = np.random.choice(np.arange(n),k, False)
    return (X[idx,:], y[idx])

def lr_sgd(X, y, k=10, gamma=0.1, eta=0.1):
    n = X.shape[0]
    x0 = np.ones((n, 1))
    X = np.hstack((x0, X))
    d = X.shape[1]
    epoch = 200
    llh = np.full(epoch, np.inf)
    w = np.zeros(d)
    for t in range(epoch):
        (Xs, ys) = sampling(X, y, k)
        g = lr_grad(Xs, ys, w, gamma)
        w = w - eta*g
        llh[t] = lr_llh(X, y, w, gamma)
    return (w, llh)

def lr_sgd_momentum(X, y, k=10, gamma=0.1, eta=0.1, rho=0.9):
    n = X.shape[0]
    x0 = np.ones((n, 1))
    X = np.hstack((x0, X))
    d = X.shape[1]
    epoch = 200
    llh = np.full(epoch, np.inf)
    w = np.zeros(d)
    v = np.zeros(d)
    for t in range(epoch):
        (Xs, ys) = sampling(X, y, k)
        g = lr_grad(Xs, ys, w, gamma)
        v = rho*v+eta*g
        w = w - v
        llh[t] = lr_llh(X, y, w, gamma)
    return (w, llh)


def lr_sgd_nag(X, y, k=10, gamma=0.1, eta=0.1, rho=0.9):
    n = X.shape[0]
    x0 = np.ones((n, 1))
    X = np.hstack((x0, X))
    d = X.shape[1]
    epoch = 200
    llh = np.full(epoch, np.inf)
    w = np.zeros(d)
    v = np.zeros(d)
    for t in range(epoch):
        (Xs, ys) = sampling(X, y, k)
        g = lr_grad(Xs, ys, w-rho*v, gamma)
        v = rho*v+eta*g
        w = w - v
        llh[t] = lr_llh(X, y, w, gamma)
    return (w, llh)    


def lr_adagrad(X, y, k=10, gamma=0.1, eta=0.1):
    n = X.shape[0]
    x0 = np.ones((n, 1))
    X = np.hstack((x0, X))
    d = X.shape[1]
    epoch = 200
    llh = np.full(epoch, np.inf)
    w = np.zeros(d)
    v = np.zeros(d)
    eps = 1e-8
    for t in range(epoch):
        (Xs, ys) = sampling(X, y, k)
        g = lr_grad(Xs, ys, w, gamma)
        v += g*g
        w = w - eta*(g/np.sqrt(v+eps))
        llh[t] = lr_llh(X, y, w, gamma)
    return (w, llh)

def lr_rmsprop(X, y, k=10, gamma=0.1, eta=0.1):
    n = X.shape[0]
    x0 = np.ones((n, 1))
    X = np.hstack((x0, X))
    d = X.shape[1]
    epoch = 200
    llh = np.full(epoch, np.inf)
    w = np.zeros(d)
    v = np.zeros(d)
    eps = 1e-8
    rho = 0.9
    for t in range(epoch):
        (Xs, ys) = sampling(X, y, k)
        g = lr_grad(Xs, ys, w, gamma)
        v = rho*v+(1-rho)*(g*g)
        w = w - eta*(g/np.sqrt(v+eps))
        llh[t] = lr_llh(X, y, w, gamma)
    return (w, llh)

# tbc

def lr_adadelta(X, y, k=10, gamma=0.1):
    n = X.shape[0]
    x0 = np.ones((n, 1))
    X = np.hstack((x0, X))
    d = X.shape[1]
    epoch = 200
    llh = np.full(epoch, np.inf)
    w = np.zeros(d)
    v = np.zeros(d)
    delta = np.zeros(d)

    eps = 1e-8
    rho = 0.9
    for t in range(epoch):
        (Xs, ys) = sampling(X, y, k)

        g = lr_grad(Xs, ys, w, gamma)
        v = rho*v+(1-rho)*(g*g)
        rms_g = np.sqrt(v+eps)


        u = rho*u+(1-rho)*delta
        rms_delta = np.sqrt(u+eps)

        w = w - rmsw/rmsg*g

        llh[t] = lr_llh(X, y, w, gamma)
    return (w, llh)

def lr_adam(X, y, k=10, gamma=0.1, eta=0.1):
    n = X.shape[0]
    x0 = np.ones((n, 1))
    X = np.hstack((x0, X))
    d = X.shape[1]
    epoch = 200
    llh = np.full(epoch, np.inf)
    w = np.zeros(d)
    v = np.zeros(d)
    eps = 1e-8
    for t in range(epoch):
        (Xs, ys) = sampling(X, y, k)
        g = lr_grad(Xs, ys, w, gamma)
        v += g*g
        w = w - eta*(g/np.sqrt(v+eps))
        llh[t] = lr_llh(X, y, w, gamma)
    return (w, llh)

def lsolve(A,B):
    return la.solve(A.T,B.T).T