import logging
import os.path
import pickle

import numpy as np
from src.network import create_sppmi_mtx,constructW_PKN
from src.network import solve_l1l2, Opt_P, wshrinkObj
from tqdm import tqdm
from  scipy.linalg import solve
import logger as l

# stACN_master
# @param expression (n,50) 特征矩阵
# @param spatial_network(n,n) 空间网络
# @param ground_truth 标签
# @return Z_all
def stACN(expression, spatial_network, gt, lamb=0.001, dim=100):
    expression = expression.T
    spatial_network = spatial_network.T

    # Data preparation and Variables init
    data = [expression, spatial_network]
    W = [None] * 2

    # 归一化
    w_dump = "./w_dump.pkl"
    if os.path.exists(w_dump):
        with open(w_dump,'rb') as  w_dump_file:
            W = pickle.load(w_dump_file)
    else :
        for i in range(2):
            # 将data[i]除以每列的平方根
            data[i] =  data[i] / np.tile(np.sqrt(np.sum(data[i] ** 2, axis=0)), (data[i].shape[0], 1))
            W[i] = create_sppmi_mtx(constructW_PKN(data[i], 10), 2)
            #W[i] = create_sppmi_mtx(data[i], 2)
        with open(w_dump,'wb') as  w_dump_file:
            pickle.dump(W,w_dump_file)
    X = W
    V = len(X)
    N = X[0].shape[0]
    Eh = [np.zeros((dim, N)) for _ in range(V)]
    Yh = [np.zeros((dim, N)) for _ in range(V)]
    Ys = [np.zeros((N, N)) for _ in range(V)]
    Zv = [np.zeros((N, N)) for _ in range(V)]
    T = [np.zeros((N, N)) for _ in range(V)]

    sX = [N, N, V]
    P = [np.zeros((dim, N)) for _ in range(V)]

    mu = 1e-4
    pho = 2
    max_mu = 1e6
    max_iter = 50
    thresh = 1e-6

    for iter_ in tqdm(range(max_iter)):
        B = np.array([]).reshape(0, Yh[0].shape[1])
        d = 0
        l.logger.info(f'[RunCSolver]iter = {iter_} calculate E')
        for i in range(V):
            Xi = X[i]
            Zvi = Zv[i]
            XZvi = np.dot(Xi,Zvi)
            XX = Xi - XZvi
            P[i] = Opt_P(Yh[i], mu, Eh[i], XX)
            # P[i] = updatePP(Yh[i], mu, Eh[i], X[i] - X[i] @ Zv[i])
            A = np.dot(P[i] , X[i])
            Zv[i] = solve(np.dot(A.T , A) + np.eye(N), np.dot(A.T , Yh[i]) / mu + np.dot(A.T , (A - Eh[i])) + T[i] - Ys[i] / mu)
            Zv[i] = (Zv[i] + Zv[i].T) / 2
            G = np.dot( P[i] , X[i]) -  np.dot(np.dot(P[i] , X[i]) , Zv[i]) + Yh[i] / mu
            B = np.vstack((B, G))
            E = solve_l1l2(B, lamb / mu)
            Eh[i] = E[d: (i+1) * dim,:]
            d += dim
        l.logger.info(f'[RunCSolver]iter = {iter_} calculate E end')

        Z_tensor = np.stack(Zv, axis=2)
        T_tensor = np.stack(T,axis=2)
        Ys_tensor = np.stack(Ys, axis=2)
        l.logger.info(f'[RunCSolver]iter = {iter_} wshrinkObj')
        t_tensor, objV = wshrinkObj(Z_tensor + 1 / mu * Ys_tensor, 1 / mu, sX, 0, 3)
        T_tensor = t_tensor.reshape(sX)
        G = []
        for i in range(V):
            Zv[i] = Z_tensor[:, :, i]
            T[i] = T_tensor[:, :, i]
            Ys[i] = Ys_tensor[:, :, i]
        for i in range(V):
            G.append( np.dot(P[i] , X[i]) - np.dot(np.dot(P[i] , X[i]) , Zv[i]) - Eh[i])
            Yh[i] = Yh[i] + mu * (G[i])
            Ys[i] = Ys[i] + mu * (Zv[i] - T[i])
        mu = min(pho * mu, max_mu)
        errp = np.zeros(V)
        errs = np.zeros(V)
        l.logger.info(f'[RunCSolver]iter = {iter_} calculate  errp errs')
        for i in range(V):
            errp[i] = np.linalg.norm(G[i], np.inf)
            errs[i] = np.linalg.norm(Zv[i] - T[i], np.inf)
        max_err = np.max(errp + errs)
        l.logger.info(f'[RunCSolver]iter = {iter_} max_err={max_err}')
        if max_err <=  thresh:
            l.logger.info(f'[RunCSolver]iter = {iter_} max_err={max_err} < {thresh} break')
            break

    Z_all = np.zeros((N, N))
    for i in range(V):
        Z_all = Z_all + (np.abs(Zv[i]) + np.abs(Zv[i].T))
    Z_all = Z_all / V

    return Z_all








