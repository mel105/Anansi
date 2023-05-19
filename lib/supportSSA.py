#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:57:33 2023

@author: mel
"""

import numpy as np


def hankelise(X):
    """
    Hankelises the matrix X, returning H(X).
    """
    L, K = X.shape
    transpose = False
    if L > K:
        # The Hankelisation below only works for matrices where L < K.
        # To Hankelise a L > K matrix, first swap L and K and tranpose X.
        # Set flag for HX to be transposed before returning.
        X = X.T
        L, K = K, L
        transpose = True

    HX = np.zeros((L, K))

    # I know this isn't very efficient...
    for m in range(L):
        for n in range(K):
            s = m+n
            if 0 <= s <= L-1:
                for l in range(0, s+1):
                    HX[m, n] += 1/(s+1)*X[l, s-l]
            elif L <= s <= K-1:
                for l in range(0, L-1):
                    HX[m, n] += 1/(L-1)*X[l, s-l]
            elif K <= s <= K+L-2:
                for l in range(s-K+1, L):
                    HX[m, n] += 1/(K+L-s-1)*X[l, s-l]
    if transpose:

        return HX.T
    else:
        return HX


def X_to_TS(X_i):
    """Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series."""

    # Reverse the column ordering of X_i

    X_rev = X_i[::-1]

    # Full credit to Mark Tolonen at https://stackoverflow.com/a/6313414 for this one:

    return np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0]+1, X_i.shape[1])])
