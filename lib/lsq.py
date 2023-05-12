#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:49:23 2022

Skript len spracuje vybrane data a autor si osaha numpy, maticovu algebru a LSQ

@author: mel
"""

import numpy as np
import csv
# import lib.plotter as plot
import lib.support as msupp
import lib.metrics as mt
import lib.model as md
import statistics as st
from tabulate import tabulate


# import matplotlib.pyplot as plt
# import pandas as pd


def processLSQ(data, lVec, W, presFitStat, listOfStations, confObj):
    """
    Funkcia moderuje spracovanie modelu pomocou metody najmensich stvorcov. Vystupom su parametre pre
    dalsie spracovanie

    Parameters
    ----------
    data : TYPE np.array
        DESCRIPTION. Matica tokov
    lVec : TYPE np.array
        DESCRIPTION. Realne hodnoty UFG
    W : TYPE np.array
        DESCRIPTION. Matica vah
    presFitStat : TYPE boolean
        DESCRIPTION. Rozhodovacie pravidlo podla ktoreho spocitam statiky po vyrovnani True, alebo nie.
    listOfStations : TYPE array
        DESCRIPTION. Zoznam spracovanych stanic
    confObj : TYPE objekt typu config
        DESCRIPTION. Zozna nastaveni

    Returns
    -------
    coef : TYPE array
        DESCRIPTION. Coeficienty, ktore by mi mali reprezentovat calc UFG
    A : TYPE array
        DESCRIPTION. Jacobi matica
    dh : TYPE array
        DESCRIPTION. Vektor oprav
    Qvv : TYPE Array
        DESCRIPTION. Kovariancna matica
    N : TYPE array
        DESCRIPTION. Matica z vyrovnania
    valEst : TYPE Array
        DESCRIPTION. Vektor vyrovnanych hodnot

    """

    # Nastavenie niektorych parametrov. NumIter je pocet aktualnych iteracii. MaxIter je pocet max
    # iteracii stopFitCrit je kriteriu, ktore je False, ak bude platit dole definovana podmienka.
    # EpsVal je hodnota ktora rozhodne o pokracovani iteracii.
    probup = confObj.getProbUp()  # up je 100P % kvantil rozdelenia
    [nrows, ncols] = data.shape
    numIter = 0
    maxIter = int(confObj.getMaxIter())
    epsVal = float(confObj.getEpsVal())
    stopFitCrit = True

    if confObj.getAddIntercept():

        # pokial chcem riesit aj absolutnu clen, teda y = beta0 + beta1*X1 + beta2*X2 + ... + eps
        initCoef = np.hstack((1.0, np.zeros(ncols) + 0.01))
    else:

        initCoef = np.zeros(ncols) + 0.01

    while stopFitCrit:

        print(f"\n  **  Iteracia (LSQ) c. {numIter}")
        numIter = numIter + 1

        # model
        # valVec = np.array(model(initCoef, data, confObj.getAddIntercept()))
        modelObj = md.model(initCoef, data, confObj.getAddIntercept())
        valVec = modelObj.estimation()
        valVec = np.array(valVec)

        resVec = lVec-valVec.reshape((-1, 1))

        # Design matrix
        # A = derivative(data, confObj.getAddIntercept())
        A = modelObj.derivative()

        # LSQ
        Qvv, coef, dh, N = lsqAlg(A, initCoef, resVec, W)

        # val est
        # valEst = np.array(model(coef, data, confObj.getAddIntercept()))
        modelObj = md.model(coef, data, confObj.getAddIntercept())
        valEst = np.array(modelObj.estimation())

        # odhad rozdielov
        # resEst = lVec-valEst.reshape((-1, 1))

        initCoef = coef

        # print(f"   ***   Norma(dh) v pripade LSQ: {np.linalg.norm(dh):.2f}")
        # print(f"   ***   Summa(val-est) v pripade LSQ: {np.sum(resEst):.2f}")

        if confObj.getAddIntercept():
            if np.linalg.norm(dh[1, :]) <= epsVal:
                stopFitCrit = False
        else:
            if np.linalg.norm(dh) <= epsVal:
                stopFitCrit = False

        if numIter == maxIter:
            stopFitCrit = False

        # statistiky porovnania modelu a dat.
        rmse, mae, bias, sumError = mt.metrics(lVec, valEst)
        # print(
        #     f"RMSE: {rmse: 0.2f}/MAE: {mae: 0.2f}/BIAS: {bias: 0.2f}/SUMA: {sumError[0]: .2f}")

        if presFitStat:

            # statistiky po vyrovnani modelu
            summary(lVec, valEst,  listOfStations, A, dh, Qvv, N, coef, probup)

    return coef, A, dh, Qvv, N, valEst


def summary(lVec, eVec, stations, A, dh, Qvv, N, coef, probup):
    """
    funkcia vrati statistiky na zaklade rozdielu medzi datami a vyrovnanym modelom

    Returns
    -------
    None.

    """

    print("\n\n       LSQ SUMMARY:      ")
    print("-----------------------------")

    print(f"Number of parameters:   {len(dh)}")
    print(f"Number of epochs:       {len(lVec)}")
    print(f"Degree of freedom:     {len(lVec) - len(dh) - 1}")

    # vektor oprav
    v = np.matmul(A, dh) - (lVec-eVec.reshape((-1, 1)))
    print(f"\n\nSuma oprav je {float(sum(v)): .2f}")

    # jednotkova stredna chyba m0
    m0 = np.sqrt(np.matmul(np.transpose(v), v) / (len(v) - len(dh)))
    print(f"Jednotkova stredna chyba je {float(m0): .2f}")

    # stredne chyby neznamych parametrov
    mC = m0 * np.sqrt(np.diag(Qvv))

    # sredne chyby oprav
    Ql = np.matmul(A, N)
    Ql = np.matmul(Ql, np.transpose(A))
    # ml = m0 * np.sqrt(np.diag(Ql))

    initCoef = coef.ravel()
    mC = mC.ravel()

    # Standardized coefficients
    std = standardCoef(coef, lVec,  A)
    std = np.array(std)

    # intervaly spolahlivosti
    islow = initCoef - 1.960*mC
    isup = initCoef + 1.960*mC

    # t-statistiky
    tstat = abs(initCoef)/mC

    # P-values
    pval = getPValues(tstat, len(lVec))

    # Kriticka hodnota T statistiky
    dof = len(lVec) - len(dh) - 1
    tcrit = critTStat(probup, dof)

    print("\nOdhad koeficientov:\n")
    print(f"Kriticka hodnota t-statistiky {tcrit: .2f}")
    # priprava tabulky na tlac
    table = []
    for i in range(len(initCoef)):
        row = [stations[i], initCoef[i], std[i], mC[i], tstat[i], pval[i], islow[i], isup[i]]
        table.append(row)

    header = ["Station", "Coef", "Standardized Coef", "Standard Deviation",
              "t-stat", "p-val", "Lower Bound", "Upper Bound"]

    print(tabulate(table,
                   headers=header,
                   tablefmt="outline"))

    with open('results.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        writer.writerow(table)

    np.savetxt("results2.csv",
               table,
               delimiter=", ",
               fmt='% s')

    # print(
    #    f" Odhady [{i}]: {initCoef[i]: .5f}  {mC[i]: .5f}  {islow[i]: .5f}  {isup[i]: .5f}\
    # {tstat[i]:.5f}  {pval[i]:.5f} -->> {stations[i-1]}")


def standardCoef(coef, lVec, A):
    """
    Funkcia vrati odhad standardizovanych koeficientov. Mali by byt v intervale od -1, 1 a poukazuju
    na fakt, ktory z regresorov ma vacsi efekt na celkovu regresiu.
    """

    y = st.stdev([y for x in lVec for y in x])
    std = []
    a, b = A.shape
    for i in range(b):
        std.append(coef[i]*(st.stdev(A[:, i])/y))  # to s dvoma indexami v coef je divne a nutne poriesit.

    return std


def getPValues(tstats, N):
    """
    double X = ( abs(_tStat[i]) ;
    double p = ;

    _pVal.push_back( 1.0 - p );

    Parameters
    ----------
    tstats : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.

    Returns
    -------
    pval : TYPE
        DESCRIPTION.

    """
    X = (abs(tstats) * (1.0 - (1.0 / (4.0 * N)))) / \
        (np.sqrt(1.0 + pow(abs(tstats), 2.0) / (2.0 * N)))
    pval = (2.0 * (1.0 - 0.5 * pow((1.0 + (0.196854 * X) + (0.115194 * pow(X, 2.0)) +
                                    (0.000344 * pow(X, 3.0)) + (0.019527 * pow(X, 4.0))),
                                   -4.0)) - 1.0)

    return 1-pval


def critTStat(probUp, dof):
    """
    Funkcia vrati kriticku hodnot t-statistiky

    Parameters
    ----------
    probUp : TYPE float
        DESCRIPTION. Pravdepodobnost na ktorej pocitame up, teda 100% kvantil rozdelenia
    dof : TYPE float
        DESCRIPTION. Stupen volnosti

    Returns
    -------
    tcrit : TYPE float
        DESCRIPTION. Odhat kritickej hodnoty. Vid Lakes, Laga, str. 17

    """

    up = getUp(probUp)

    tcrit = up * (1 + (1/(4*dof))*(1+up*up) + (1/(96*dof*dof)) * (3 + 16*up*up + 5 * up*up*up*up*up))

    return tcrit


def getUp(prob):
    """
    Funkcia vrati 100% kvantil rozdelenia na zaklade volby pravdepodobnosti

    Parameters
    ----------
    prob : TYPE
        DESCRIPTION.

    Returns
    -------
    up : TYPE
        DESCRIPTION.

    """
    if prob == 0.683:
        up = 0.476104  # approx 1*sigma
    elif prob == 0.950:
        up = 1.644854
    elif prob == 0.95:
        up = 1.695398  # approx 2*sigma
    elif prob == 0.975:
        up = 1.959964
    elif prob == 0.990:
        up = 2.326348
    elif prob == 0.995:
        up = 2.575829
    elif prob == 0.997:
        up = 2.747781  # approx 3*sigma
    else:
        up = 3.090232

    return up


def lsqAlg(A, initCoef0, lvec, W):
    """
    Funkcia vrati kovariancnu maticu, opravy a vyrovnane parametre

    Returns
    -------
    None.

    """

    A = np.array(A)

    # LSQ
    At = np.transpose(A)
    AtW = np.matmul(At, W)
    N = np.matmul(AtW, A)
    N = N.astype(float)
    Qvv = np.linalg.inv(N)
    AtL = np.matmul(AtW, lvec)

    # coef
    dh = np.matmul(Qvv, AtL)

    initCoef = initCoef0 + dh.T

    return Qvv, initCoef, dh, N


"""
def derivative(inp, inter):
    "" "
     Funkcia vrati matici dizajnu

     Returns
     -------
     None.

    "" "

    if inter:
        # v pripade, ak by som chcel modelovat aj abs. clek. V jacobi matici, prvy stlpec jednickovy.
        A = np.c_[np.ones(len(inp)), inp]
    else:
        # inak
        A = inp

    return A


def model(coef, inp, inter):
    "" "
     Funkcia obsahuje model, ktory fitujem. Na vstupe sa nachadzaju jak odhadnute koeficienty, tak aj vstupne
     data tokov. Vysledkom je model strat, ktory ziskam prenasobenim koeficientov a tokov.

    Returns
    -------
    None.

    "" "

    # kontrola, ze ci coef je slpcovy alebo riadkovy vektor. Potrebujem stlpcovy
    msize = coef.shape
    if msize[0] == 1:
        coef = coef.T

    # prenasobenie koeficientov tokmi: mu + A*kA + B*kB + C*kC = model
    val = []

    if inter:

        for i in range(len(inp)):
            tmp = float(coef[0])
            for j in range(len(coef)-1):
                # print(i, j, tmp, coef[0])
                tmp += float(coef[j+1])*float(inp[i, j])

            val.append(tmp)
    else:

        for i in range(len(inp)):
            tmp = 0
            for j in range(len(coef)):
                # print(" f.2%   f.2%", i, j)
                tmp += coef[j]*float(inp[i, j])

            val.append(tmp)

    return val
"""
