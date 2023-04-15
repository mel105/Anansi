#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:07:20 2023

@author: mel
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from tabulate import tabulate
# import lib.plotter as plot


def metrics(refVec, estVec):
    """
    Funkcia vypise suhrn odhadovanych metrik, ktore sa tykaju pmErrornosti medzi modelom a orig
    datmi

    Parameters
    ----------
    refVec : TYPE
        DESCRIPTION. vektor referencnych dat
    estVec : TYPE
        DESCRIPTION. vektor odhadovanych dat

    Returns
    -------
    int
        DESCRIPTION.

    """

    mError = refVec - estVec

    # ###### STATISTIKY, KTORE SA TYKAJU OPISU CHYB MEDZI REF. A ODHADOVANYM SUBOROM DAT ######
    # RMSE
    squaredError = mError ** 2
    meanSquaredError = squaredError.mean()
    rmse = np.sqrt(meanSquaredError)

    # MAE
    absError = abs(mError)
    mae = absError.mean()

    # BIAS
    bias = mError.mean()

    # Suma chyb
    sumaError = sum(mError)

    # ##### STATISTIKY, KTORE SA TYKAJU KVALITY APROXIMACIE ############

    # RSE, CD
    # refVecMean = refVec.mean()
    # ST = .0
    # SC = .0
    # for idx, i in enumerate(refVec):
    #     ST += np.power(estVec[idx]-refVecMean, 2)
    #     SC += np.power(refVec[idx]-refVecMean, 2)

    # CD = 1-ST/SC
    # R, _ = spearmanr(refVec, estVec)
    # R22 = r2_score(refVec, estVec)
    # R2 = np.power(R, 2)

    # ######## VYPIS NA PLOCHU. SUBOR METRIK OBSAHUJE AK KVANTILY ROZDELENIA ##########
    # tabulka na plochu
    print("\nOdhad metrik:\n")

    table = [
        ["MIN", np.quantile(mError, 0.00)],
        ["Q25", np.quantile(mError, 0.25)],
        ["Q50", np.quantile(mError, 0.50)],
        ["Q75", np.quantile(mError, 0.75)],
        ["MAX", np.quantile(mError, 1.00)],
        ["RMS", rmse],
        ["MAE", mae],
        ["BIAS", bias],
        ["SUME", sumaError]]

    print(tabulate(table, headers=["Statistics", "Estimation"], tablefmt="outline", floatfmt=".7f"))

    return rmse, mae, bias, sumaError
