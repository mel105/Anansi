#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:07:20 2023

@author: mel
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import lib.plotter as plot


def metrics(lVec, eVec):
    """
    Funkcia na plochu vypise suhrn odhadovanych metrik, ktore sa tykaju presnosti medzi modelom a orig
    datmi

    Parameters
    ----------
    lVec : TYPE
        DESCRIPTION.
    eVec : TYPE
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """

    # statistiky, tykajuce sa rozdielu modelu a merani
    # res = lVec-eVec.reshape((-1, 1))
    res = lVec-eVec

    print("\n\n         RESIDUALS:      ")
    print("-----------------------------")
    print(f"MIN: {np.quantile(res, 0.00):.2f}")
    print(f"Q25: {np.quantile(res, 0.25):.2f}")
    print(f"MED: {np.quantile(res, 0.50):.2f}")
    print(f"Q75: {np.quantile(res, 0.75):.2f}")
    print(f"MAX: {np.quantile(res, 1.00):.2f}")

    # RSE, CD
    lVecMean = lVec.mean()
    ST = .0
    SC = .0
    for idx, i in enumerate(lVec):
        ST += np.power(eVec[idx]-lVecMean, 2)
        SC += np.power(lVec[idx]-lVecMean, 2)

    CD = 1-ST/SC
    R, _ = spearmanr(lVec, eVec)
    R22 = r2_score(lVec, eVec)
    R2 = np.power(R, 2)

    print(f"R: {R:0.3f}")
    print(f"R2: {R2:0.3f}")
    print(f"R22: {R22:0.3f}")
#    print(f"D: {CD[0]:0.3f}")

    # Scatter graf
    plot.plotScatter(lVec, eVec)

    # Histogram rozdielov
    print("\n\n         HISTOGRAM:      ")
    print("-----------------------------")
    # plot.plotHistogram(res)
    print("OK")
    return 0
