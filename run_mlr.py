#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 08:11:04 2023

@author: mel
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import lib.support as sp
import lib.decoder as dc
import lib.config as cfg

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def testMLR():
    """
    Funkcia otestuje MLR pomocou python packagies

    Returns
    -------
    None.

    """

    print("TEST MLR Algoritmu")

    # Nacitanie koniguracneho suboru
    confObj = cfg.config()

    # Nastavenie adresarov, kam sa budu ukladat obrazky/subory CSV (neskor treba textove protokoly)
    sp.checkFolder(confObj.getOutLocalPath())
    sp.checkFolder(confObj.getOutLocalPath()+"/"+confObj.getFigFolderName())
    sp.checkFolder(confObj.getOutLocalPath()+"/"+confObj.getCsvFolderName())

    # Nacitanie dat
    fileName = confObj.getInpFileName()
    filePath = confObj.getInpFilePath()
    decObj = dc.decoder(fileName, filePath, confObj)

    # Rozdelenie dat na zavisle a nezavisle veliciny. Tie su definovane v konfiguraku
    listOfStations = confObj.getOutStations()

    df = decObj.getDF()

    dt = sp.fillDataContainer(df, listOfStations)
    X = pd.DataFrame(data=dt, columns=listOfStations)

    print("Matica korelacnych koeficientov")
    print(X.corr())

    y = df["TOTAL NB"]
    print("Hustota rozdelenia sledovaneho parametru")
    sns.distplot(y)
    plt.show()

    # with statsmodels
    x = sm.add_constant(X)  # adding a constant

    model = sm.OLS(y, x).fit()
    predictions = model.predict(x)

    print_model = model.summary()
    print(print_model)

    # MLR
    # Splitting the dataset into the Training set and Test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Fitting Multiple Linear Regression to the Training set
    # regressor = LinearRegression()
    # regressor.fit(X_train, y_train)

    # Predicting the Test set results
    # y_pred = regressor.predict(X_test)

    # score = r2_score(y_test, y_pred)

    # print("R2 SCORE: ", score)

    return confObj, decObj, X, y


def compute_vif(df, considered_features):

    X = df[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1

    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable'] != 'intercept']

    return vif


# Spustenie spracovania dat
if __name__ == "__main__":

    conf, dec, dataX, dataY = testMLR()
