import numpy as np
from numpy.linalg import svd
# from scipy.linalg import svd
from tqdm import trange
import scipy.sparse as sp

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

    sppmi = G.copy()

    # Use a loop to calculate Wij*W/(di*dj)
    col, row, weights = sp.find(G)
    for i in range(len(col)):
        score = np.log(weights[i] * W / (node_degrees2[col[i]] * node_degrees[row[i]])) - np.log(k)
        sppmi[col[i], row[i]] = max(score, 0)

    return sppmi


def softth(F, lambda_val):
    temp = F.copy()
    U, S, Vt = np.linalg.svd(temp, full_matrices=False)
    Vt = Vt.T

    svp = len(np.flatnonzero(S > lambda_val))
    # svp = np.count_nonzero(S > lambda_val)

    diagS = np.maximum(0, S - lambda_val)

    if svp < 1:
        svp = 1

    E = U[:, :svp] @ np.diag(diagS[:svp]) @ Vt[:, 0: svp].T
    return E


def sparse_self_representation(x, init_w, alpha=1, beta=1):
    # x \in R^{d \times n}
    max_epoch = 100
    n = x.shape[1]
    T1 = np.zeros((n, n))

    C = init_w.copy()
    J1 = C.copy()
    mu = 50
    D = np.diag(np.sum(init_w, axis=0))

    epoch_iter = trange(max_epoch)
    for epoch in epoch_iter:
        # 更新 C 矩阵
        C = C * ((x.T @ x + mu * (J1 - np.diag(np.diag(J1))) - T1 + beta * init_w @ C) /
                 (x.T @ x @ C + mu * C + beta * D @ C))
        C[np.isnan(C)] = 0
        C = C - np.diag(np.diag(C))
        # 计算 J1 矩阵
        J1 = np.array(soft_numpy(C + T1 / mu, alpha / mu))
        J1 = J1 - np.diag(np.diag(J1))
        # 更新 T1 矩阵
        T1 = T1 + mu * (C - J1)

        # 计算误差
        err = np.linalg.norm(x - x @ C, 'fro')
        if err < 1e-2:
            break

        epoch_iter.set_description(f"# Epoch {epoch}, loss: {err.item():.3f}")
    C = 0.5 * (np.abs(C) + np.abs(C.T))
    return C


def solve_l1l2(W, lamb):
    n = W.shape[0]
    E = W.copy()

    for i in range(n):
        E[:, i] = solve_l2(W[:, i], lamb)
    return E


def solve_l2(w, lamb):
    nw = np.linalg.norm(w)

    if nw > lamb:
        x = (nw - lamb) * w / nw
    else:
        x = np.zeros_like(w)
    return x


def Opt_P(Y, mu, A, X):
    G = X.T
    #Q = (A - Y / mu).T
    Q = (Y / mu-A ).T
    W = np.dot(G.T, Q) + np.finfo(float).eps
    U, S, Vt = svd(W, full_matrices=False)
    # U, S, Vt = svd(W, 0)
    PT = np.dot(U, Vt)
    P = PT.T
    return P


# def constructW_PKN(X, k, issymmetric=True):
#     if issymmetric:
#         issymmetric = 1
#     if k is None:
#         k = 5
#     dim, n = X.shape
#     D = L2_distance_1(X, X)
#     idx = np.argsort(D, axis=1)  # sort each row
#     W = np.zeros((n, n))
#     for i in range(n):
#         id = idx[i, 1:k + 1]
#         di = D[i, id]
#         W[i, id] = (di[k] - di) / (k * di[k] - np.sum(di[:k]) + np.finfo(float).eps)
#     if issymmetric == 1:
#         W = (W + W.T) / 2
#     return W


def constructW_PKN(X, k=5, issymmetric=1):
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
    #D = cdist(X.T, X.T, metric='euclidean')
    D = L2_distance_1(X,X)
    idx = np.argsort(D, axis=1)  # sort each row

    W = np.zeros((n, n))
    for i in range(n):
        id = idx[i, 1:k + 2]
        di = D[i, id]
        W[i, id] = (di[k] - di) / (k * di[k] - np.sum(di[:k]) + np.finfo(float).eps)

    if issymmetric == 1:
        W = (W + W.T) / 2

    return W


def L2_distance_1(a, b):
    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros((1, a.shape[1]))])
        b = np.vstack([b, np.zeros((1, b.shape[1]))])
    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    ab = np.dot(a.T, b)
    d = np.tile(aa[:, np.newaxis], (1, bb.shape[0])) + np.tile(bb, (aa.shape[0], 1)) - 2 * ab
    d = np.real(d)
    d = np.maximum(d, 0)
    return d


def wshrinkObj(x, rho, sX, isWeight, mode):
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

    if n3 % 2 == 0:
        endValue = np.int16(np.floor(n3/2)+ 1)
        Yhat = np.real(Yhat)
        for i in range(1, endValue+1):
            uhat, shat, vhat = svd(Yhat[:, :, i-1], full_matrices=False)

            if isWeight:
                weight = C / (np.diag(shat) + np.finfo(float).eps)
                tau = rho * weight
                shat = soft(shat, np.diag(tau))
            else:
                tau = rho
                shat = np.maximum(shat - tau, 0)

            objV += np.sum(shat)
            Yhat[:, :, i-1] = np.dot(np.dot(uhat, np.diag(shat)), vhat.T)
            if i > 1:
                Yhat[:, :, n3 - i+1] = np.dot(np.dot(np.conj(uhat), np.diag(shat)), np.conj(vhat).T)
                objV += np.sum(shat)

        uhat, shat, vhat = svd(Yhat[:, :, endValue], full_matrices=False)

        if isWeight:
            weight = C / (np.diag(shat) + np.finfo(float).eps)
            tau = rho * weight
            shat = soft(shat, np.diag(tau))
        else:
            tau = rho
            shat = np.maximum(shat - tau, 0)

        objV += np.sum(shat)
        Yhat[:, :, endValue] = np.dot(np.dot(uhat, np.diag(shat)), vhat.T)
    else:
        endValue = np.int16(np.floor(n3/2) + 1)
        Yhat = np.real(Yhat)
        for i in range(1, endValue+1):
            uhat, shat, vhat = svd(Yhat[:, :, i-1], full_matrices=False)
            if isWeight:
                weight = C / (np.diag(shat) + np.finfo(float).eps)
                tau = rho * weight
                shat = soft(shat, np.diag(tau))
            else:
                tau = rho
                shat = np.maximum(shat - tau, 0)
            objV += np.sum(shat)
            Yhat[:, :, i-1] = np.dot(np.dot(uhat, np.diag(shat)), vhat.T)
            if i > 1:
                Yhat[:, :, n3 - i+1] = np.dot(np.dot(np.conj(uhat), np.diag(shat)), np.conj(vhat).T)
                objV += np.sum(shat)

    Y = np.real(np.fft.ifft(Yhat, axis=2))

    if mode == 1:
        X = np.fft.ifft(Y, axis=2)
    elif mode == 3:
        X = np.moveaxis(Y, -1, 0)
    else:
        X = Y

    x = X.flatten()

    return x, objV
