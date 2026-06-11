# Source - https://stackoverflow.com/a/29017543
# Posted by Arun Chaganty
# Retrieved 2026-03-24, License - CC BY-SA 3.0

#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Routines for simultaneous diagonalization
Arun Chaganty <arunchaganty@gmail.com>
"""

from numba import njit
import numpy as np

@njit
def givens_double_rotate_inplace(M, i, j, c, s):
    n = M.shape[0]

    # lignes
    for k in range(n):
        Mi = M[i, k]
        Mj = M[j, k]
        M[i, k] = c * Mi + s * Mj
        M[j, k] = c * Mj - s * Mi

    # colonnes
    for k in range(n):
        Mi = M[k, i]
        Mj = M[k, j]
        M[k, i] = c * Mi + s * Mj
        M[k, j] = c * Mj - s * Mi


@njit
def givens_rotate_inplace(R, i, j, c, s):
    n = R.shape[0]
    for k in range(n):
        Ri = R[i, k]
        Rj = R[j, k]
        R[i, k] = c * Ri + s * Rj
        R[j, k] = c * Rj - s * Ri


@njit
def joint_diag(Ms, sweeps=10, eps=1e-1):
    k, n, _ = Ms.shape

    R = np.eye(n)

    for _ in range(sweeps):
        done = True

        for i in range(n):
            for j in range(i + 1, n):

                g0 = 0.0
                g1 = 0.0

                for m in range(k):
                    M = Ms[m]
                    a = M[i, i] - M[j, j]
                    b = M[i, j] + M[j, i]
                    g0 += a * a
                    g1 += a * b

                g0 /= k
                g1 /= k

                t_on = g0 - g1
                t_off = 2.0 * g1

                denom = np.sqrt(t_on * t_on + t_off * t_off) + 1e-12
                theta = 0.5 * np.arctan2(t_off, t_on + denom)

                c = np.cos(theta)
                s = np.sin(theta)

                if abs(s) > eps:
                    done = False

                    for m in range(k):
                        givens_double_rotate_inplace(Ms[m], i, j, c, s)

                    givens_rotate_inplace(R, i, j, c, s)

        if done:
            break

    # --- sortie ---
    R = R.T

    L = np.zeros((n, k))
    err = 0.0

    for m in range(k):
        M = Ms[m]

        # diag
        for i in range(n):
            L[i, m] = M[i, i]

        # erreur (off-diagonal)
        for i in range(n):
            for j in range(n):
                if i != j:
                    err += M[i, j] * M[i, j]

    return R, L, err
