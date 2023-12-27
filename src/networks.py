import numpy as np
from tqdm import trange
from numpy.linalg import svd
from scipy.spatial.distance import cdist


def soft_numpy(x, T):
    if np.sum(np.abs(T)) == 0.:
        y = x
    else:
        y = np.maximum(np.abs(x) - T, 0.)
        y = np.sign(x) * y
    return y


def soft(x, T):
    if np.sum(np.abs(T)) == 0.:
        y = x
    else:
        y = np.maximum(np.abs(x) - T, 0.)
        y = np.sign(x) * y
    return y


def create_sppmi_mtx(G, k):
    node_degrees = np.array(G.sum(axis=0)).flatten()
    node_degrees2 = np.array(G.sum(axis=1)).flatten()
    W = np.sum(node_degrees)

    sppmi = np.zeros_like(G).astype(float)
    row, col = np.nonzero(G > 0)

    for index in range(len(col)):
        i = row[index]
        j = col[index]
        score = np.log(G[i][j] * W / (node_degrees2[row[index]] * node_degrees[col[index]])) - np.log(k)
        sppmi[row[index], col[index]] = max(score, 0.0)

    return sppmi


def solve_l1l2(W, lamb):
    n = W.shape[0]
    E = W.copy()

    for i in range(n):
        E[i, :] = solve_l2(W[i, :], lamb)
        # print(E[:, i])
    return E


def solve_l2(w, lamb):
    nw = np.linalg.norm(w)
    # print(nw)
    if nw > lamb:
        # print(w)
        x = (nw - lamb) * w / nw
    else:
        x = np.zeros_like(w)
    return x


def opt_p(Y, mu, A, X):
    G = X.T
    Q = (A - Y / mu).T
    # Q = (Y / mu-A ).T
    W = np.dot(G.T, Q) + np.finfo(float).eps
    U, S, Vt = svd(W, full_matrices=False)
    # U, S, Vt = svd(W, 0)
    PT = np.dot(U, Vt)
    P = PT.T
    return P


def construct_w_pkn(X, k=5, issymmetric=1):
    """
    Construct similarity matrix W using the PKN algorithm.

    Parameters:
    - X: Each column is a data point.
    - k: Number of neighbors.
    - issymmetric: Set W = (W + W')/2 if issymmetric=1.

    Returns:
    - W: Similarity matrix.
    """
    dim, n = X.shape
    D = cdist(X.T, X.T, metric='euclidean') ** 2

    idx = np.argsort(D, axis=1)  # sort each row

    W = np.zeros((n, n))
    for i in range(n):
        id = idx[i, 1:k + 2]
        di = D[i, id]
        W[i, id] = (di[k] - di) / (k * di[k] - np.sum(di[:k]) + np.finfo(float).eps)

    if issymmetric == 1:
        W = (W + W.T) / 2

    return W


def wshrink_obj(x, rho, sX, isWeight, mode, output_path="output", is_reload=False):
    if isWeight == 1:
        C = np.sqrt(sX[2] * sX[1])
    if mode is None:
        mode = 1
    X = x.reshape(sX)
    if mode == 1:
        Y = np.swapaxes(X, 0, 2)
    elif mode == 3:
        Y = np.moveaxis(X, 0, -1)
    else:
        Y = X

    Yhat = np.fft.fft(Y, axis=2)
    objV = 0

    if mode == 1:
        n3 = sX[1]
    elif mode == 3:
        n3 = sX[0]
    else:
        n3 = sX[2]

    endValue = np.int16(np.floor(n3 / 2) + 1)

    for i in range(endValue):
        uhat, shat, vhat = svd((Yhat[:, :, i]), full_matrices=False)
        if isWeight:
            weight = C / (np.diag(shat) + np.finfo(float).eps)
            tau = rho * weight
            shat = soft(shat, np.diag(tau))
        else:
            tau = rho
            shat = np.maximum(shat - tau, 0)
        objV += np.sum(shat)
        Yhat[:, :, i] = np.dot(np.dot(uhat, np.diag(shat)), vhat)
        if i > 1:
            Yhat[:, :, n3 - i] = np.dot(np.dot(np.conj(uhat), np.diag(shat)), np.conj(vhat))
            objV += np.sum(shat)

    Y = np.fft.ifft(Yhat, axis=2)
    Y = np.real(Y)

    if mode == 1:
        X = np.fft.ifft(Y, axis=2)
    elif mode == 3:
        X = np.moveaxis(Y, -1, 0)
    else:
        X = Y

    x = X.flatten()
    return x, objV
