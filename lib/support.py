#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 08:40:42 2023

@author: mel
"""

import pandas as pd
import numpy as np


def fillDataContainer(df, listOfStations):
    """
    Funkcia vrati pole vybranych dat na zaklade listOfStation. Neskor prerobit na zaklade konfiguraku a
    aby tam bolo znamienko. Takze namiesto listOfStation bude skor slovnik, key = stanice, val = -1,1.

    Dostal som toto upozronenie:
    /home/mel/Dokumenty/M/NET4GAS/LSQ_spracovanie_pritokov/support.py:32: FutureWarning: Using the
    level keyword in DataFrame and Series aggregations is deprecated and will be removed in a future
    version. Use groupby instead. df.sum(level=1) should use df.groupby(level=1).sum().
          data = pd.concat(d, axis=1).sum(axis=1, level=0)

    Parameters
    ----------
    df : TYPE dataframe
        DESCRIPTION. Kontainer nacitanych dat
    listOfStations : TYPE np.array
        DESCRIPTION. Zoznam stanic

    Returns
    -------
    data : TYPE numpy.ndarray
        DESCRIPTION. Vysledne pole dat, ktore budem spracuvavat

    """

    # vyroba kontajnera. Z historickych dovodov zatial pracujem s polom
    d = {i: df[[i]] for i in listOfStations}
    data = pd.concat(d, axis=1).groupby(axis=1, level=0).sum()

    # Z historickych dovodov pracujem s polom
    data = data.to_numpy()

    # Prazdny string nahradim cislom nula, ak take pole v datach existuje
    data[np.where(data == " ")] == 0.0

    # stara cast kodu, kedy som predpokladal, ze calc ufg sa pocita exit-enter. Po novom to nerobim
    """
    idx = 0
    for i in data:
        data[i] = data[i]*listOfSigns[idx]
        idx = idx+1
    """

    return data


def calcWeights(unt, df):
    """
    Funkcia spocita takzvanu maticu vah a vysledkom je hlavna diagonala tejto matice. Tu je priestor
    na hranie sa v zmysle, ako jednotlive vahy definovat. Momentalne na vystupe dostaneme vahy
    definovane ako odmocninca rozdielu Odchod-Prichod alebo jednickovy vektor.

    Parameters
    ----------
    unt : TYPE boolean
        DESCRIPTION. True, mam sa venovat vaham. False, vysledkom bude vektor jedniciek
    df : TYPE dataframe
        DESCRIPTION. data. Zatial plny kontajner, pretoze na zaklade nejakeho vyskumu by som sa mohol
        rozhodnut pre iny typ vypoctu. Ale ak ponecham koncept Off-Int, potom by sa oplatilo tento
        vektor mat na vstupe.

    Returns
    -------
    weightsVec : TYPE numpy.array
        DESCRIPTION. Vektor vah

    """

    if unt:

        wVec = np.sqrt(abs(df.Offtakes-df.Intakes))
    else:

        wVec = np.ones(df.shape[0])

    return np.diag(wVec)


def metrics(refVec, estVec):
    """
    Funkcia vrati zakladne metriky pre porovnanie modelu vs. merania

    Parameters
    ----------
    refVec : TYPE numpy.mdarray
        DESCRIPTION. Referencne hodnoty
    estVec : TYPE numpy.mdarray
        DESCRIPTION. Odhadnute hodnty modelu

    Returns
    -------
    rmse : TYPE float
        DESCRIPTION. root mean squared error
    mae : TYPE float
        DESCRIPTION. mean average error
    bias : TYPE float
        DESCRIPTION. systematicka chyba
    """

    mError = refVec - estVec

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

    return rmse, mae, bias, sumaError


def reduceDataArray(coef, data, listOfStations, confObj):
    """

    Funkcia vrati redukovane pole Data. Redukovane je o hodnoty, ktore nesplnia zadane podmienky.
    Zadane podmienky su v skripte uvedene fixne. Je mozne neskor zvazit moznost, zeby boli sucastou
    konfiguracneho suboru. Skript funguje asi na tomto principe: Vo vektore coef si zistim najvacsi
    koeficient. Ak je koeficient vacsi ako fix podmienka, potom si zisti index tohto koeficientu a z
    pola data odstran vektor, ktory sa nachadza na rovnakom indexe. Vysledkom je stop kriterium, ktore
    je false, ak  vsetky koeficienty vyhovuju pozadovanej fix podmienke. Dalej je to novy vektor
    koeficientov a upravene pole dat.

    Parameters
    ----------
    coef : TYPE numpy.ndarray
        DESCRIPTION. Vektor, ktory obsajuje analyzovane koeficienty
    data : TYPE numpy.ndarray
        DESCRIPTION. Pole dat, ktore pochadzaju z merani prietokov plynu na vybranych staniciach.
    listOfStations : TYPE numpy.ndarray
        DESCRIPTION. Vektor stanic
    conf : TYPE objekt typu config
        DESCRIPTION. Nastavenie

    Returns
    -------
    stopDataCrit : TYPE boolean
        DESCRIPTION. Ak False, potom vektor Coef uz obsahjuje koeficienty, ktore splnuju nasu podmienku
    coefAnal : TYPE numpy.ndarray
        DESCRIPTION. Redukovane pole koeficientov
    dataOut : TYPE numpy.ndarray
        DESCRIPTION. Redukovane pole tokov.
    """

    coef = coef.T
    if confObj.getAddIntercept():
        maxElement = np.amax(abs(coef[1:]))
    else:
        maxElement = np.amax(abs(coef))

    maxIdx = []

    if maxElement >= confObj.getLimitRelativeUfG():
        maxIdx = np.where(abs(coef) == maxElement)

        coefOut = np.delete(coef, maxIdx[0])
        listOfStationsOut = np.delete(listOfStations, maxIdx[0])
        dataOut = np.delete(data, maxIdx[0], axis=1)

        stop = True
    else:
        maxIdx = [999]
        coefOut = coef
        listOfStationsOut = listOfStations
        dataOut = data
        stop = False

    return stop, coefOut, dataOut, maxIdx[0], maxElement, listOfStationsOut
