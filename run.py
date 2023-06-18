#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 08:11:04 2023

Postupne by tento skript ma zastresit tieto ukony:
    0. spracovat data a na zaklade navrhnuteho modelu sumarizovat vysledky
    1. vyrobit html report (html stranka HPS stanic, ina stranka pre DSS, DCC atp. nejake prehlady
       atp.)
    2. vyrobit forecasting: na zaklade zadanych tokov predpoved pre dalsie roky a plus neistoty
    3. spracovat kalkulacku zasobnikov: preprobeny excel do python kodu (mozno samostatny program)
    4. analyzovat efekt teploty na akumulaciu. Minimalne 3D mapa teploty v -x metrov pod povrchom.
       (mozno samostatny program)
    5. Pripadne venovat sa inym analyzam podla potreby

@author: mel
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

import lib.config as cfg
import lib.processModel as prc
import lib.smoothSeries as smt
import lib.metrics as mt
import lib.processUfg as ufg
import lib.decoder as dc
import lib.validation as vl
import lib.descriptive as ds
import lib.charts as cha
import lib.visualRelations as rel
import lib.visualGroups as grp
import lib.support as sp
import lib.supportSSA as spssa
import lib.SSA as ssa

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


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
    # vsetky tieto remove stanice vychadzaju z predpokladu overenia multikolinearity. Pred vyhodenim Barousova
    # presnost modelu je 62,69%  a condition number 4,14e+9

    df = decObj.getDF()
    dt = sp.fillDataContainer(df, listOfStations)
    X = pd.DataFrame(data=dt, columns=listOfStations)
    y = df["TOTAL NB"]

    # set figure size
    # plt.figure(figsize=(10, 7))

    # Generate a mask to onlyshow the bottom triangle
    # mask = np.triu(np.ones_like(X.corr(), dtype=bool))

    # generate heatmap
    # sns.heatmap(X.corr(), annot=True, mask=mask, vmin=-1, vmax=1)
    # plt.title('Correlation Coefficient Of Predictors')
    # plt.show()

    # Clustermap
    # plt.figure(figsize=(40, 40))
    # sns.clustermap(X.corr())
    # plt.show()
    plt.figure(figsize=(40, 40))
    sns.pairplot(X, kind="scatter", diag_kind="kde", height=1)
    plt.savefig("pairplot.png")
    # plt.show()
    # Overenie multikolinearity
    cv = compute_vif(X, listOfStations).sort_values('VIF', ascending=False)
    print(cv)

    # MLR pomocou sklear
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regObj = linear_model.LinearRegression()
    regObj.fit(X_train, y_train)
    y_pred = regObj.predict(X_test)
    accuracy = r2_score(y_test, y_pred)*100
    print(" Accuracy of the model is %.2f" % accuracy)

    # MLR pomocou statsmodels
    X = sm.add_constant(X)  # adding a constant

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    print_model = model.summary()
    print(print_model)

    return decObj, X, regObj, predictions, cv


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


def testSSA():
    """
    Funkcia otestuje SSA pre ucel vyhladenia UFG

    Returns
    -------
    None.

    """

    print("TEST SSA Algoritmu")

    # Vygenerovanie umelej casovej rady
    N = 200  # The number of time 'moments' in our toy series
    t = np.arange(0, N)
    trend = 0.001 * (t - 100)**2
    p1, p2 = 20, 30
    periodic1 = 2 * np.sin(2*math.pi*t/p1)
    periodic2 = 0.75 * np.sin(2*math.pi*t/p2)

    np.random.seed(123)  # So we generate the same noisy time series every time.
    noise = 2 * (np.random.rand(N) - 0.5)
    F = trend + periodic1 + periodic2 + noise

    # vykreslenie casovej rady
    # cha.generalPlot(t, F, trend, periodic1, periodic2, noise)

    # Trajectory matrix
    L = 70  # The window length.

    K = N - L + 1  # The number of columns in the trajectory matrix.
    # Create the trajectory matrix by pulling the relevant subseries of F, and stacking them as columns.
    X = np.column_stack([F[i:i+L] for i in range(0, K)])
    # Note: the i+L above gives us up to i+L-1, as numpy array upper bounds are exclusive.

    # cha.trajectoryMatrix(X)

    # Decomposition of trajectory matrix
    d = np.linalg.matrix_rank(X)  # The intrinsic dimensionality of the trajectory space.

    # For those interested in how to code up an SVD calculation, Numerical Recipes in Fortran 77
    # has you covered: http://www.aip.de/groups/soe/local/numres/bookfpdf/f2-6.pdf
    # Thankfully, we'll leave the actual SVD calculation to NumPy.
    U, Sigma, V = np.linalg.svd(X)
    V = V.T  # Note: the SVD routine returns V^T, not V, so I'll tranpose it back here. This may seem pointles
    # but I'll treat the Python representation of V consistently with the mathematical notation in this notebo

    # Calculate the elementary matrices of X, storing them in a multidimensional NumPy array.
    # This requires calculating sigma_i * U_i * (V_i)^T for each i, or sigma_i * outer_product(U_i, V_i).
    # Note that Sigma is a 1D array of singular values, instead of the full L x K diagonal matrix.
    X_elem = np.array([Sigma[i] * np.outer(U[:, i], V[:, i]) for i in range(0, d)])

    # Quick sanity check: the sum of all elementary matrices in X_elm should be equal to X, to within a
    # *very small* tolerance:
    if not np.allclose(X, X_elem.sum(axis=0), atol=1e-10):
        print("WARNING: The sum of X's elementary matrices is not equal to X!")

    # zobrazenie prvych 12 elem matic
    n = min(12, d)

    # cha.elementaryMatrices(n, X_elem)

    sigma_sumsq = (Sigma**2).sum()

    # cha.contributionPlot(Sigma, sigma_sumsq)

    # Time series rekonstruction
    # cha.hankeliseMatrices(n, X_elem)

    # zobrazenie prvych n komponent
    # cha.plotNcomponents(t, F, X_elem, n)

    # zlepenie komponent dokopy: Tu to bude chciet nejak zautomatizovat
    # Assemble the grouped components of the time series.
    F_trend = spssa.X_to_TS(X_elem[[0, 1, 6]].sum(axis=0))
    F_periodic1 = spssa.X_to_TS(X_elem[[2, 3]].sum(axis=0))
    F_periodic2 = spssa.X_to_TS(X_elem[[4, 5]].sum(axis=0))
    F_noise = spssa.X_to_TS(X_elem[7:].sum(axis=0))

    # cha.generalReconstruction(t, F, F_trend, F_periodic1, F_periodic2, F_noise)

    # A list of tuples so we can create the next plot with a loop.
    components = [("Trend", trend, F_trend),
                  ("Periodic 1", periodic1, F_periodic1),
                  ("Periodic 2", periodic2, F_periodic2),
                  ("Noise", noise, F_noise)]

    # cha.plotComponents(t, F, components)

    # Tu je cast kodu, kde je pouzita trieda SSA a kde sa diskutuje automaticke
    # rozodnutie, ktore komponenty ako grupovat.
    L = 60
    F_ssa = ssa.SSA(F, L)

    cmpdf = F_ssa.components_to_df()

    # F_ssa.orig_TS.plot(alpha=0.4)
    # plt.xlabel("$t$")
    # plt.ylabel(r"$\tilde{F}_i(t)$")
    # plt.title(r"$L={0}$ for the Toy Time Series".format(L))

    # plt.show()

    mtitle = "W-Correlation for analysed Time Series, L=60"
    F_ssa.plot_wcorr(mtitle)

    # RECONSTRUCTION
    sig = F_ssa.reconstruct(slice(0, 7))
    noi = F_ssa.reconstruct(slice(8, 60))
    org = F_ssa.orig_TS

    cha.ssaPlot(t, org, sig, noi)


def run():

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

    # Zobrazenie orig strat a vyhladenych strat, ak SSA je v nastaveni zvolena ako smoothed method.
    if confObj.getSmoothingMethod() == "SSA":
        cha.multilineChart(pd.DataFrame({"DATE": decObj.getDF()["DATE"],
                                         "Original UfG": decObj.getDF()["TOTAL NB ORIG"],
                                         "Smoothed UfG": decObj.getDF()["TOTAL NB"]}),
                           title="Original and Smoothed real UfG comparision",
                           xLabel="t (day)", yLabel="UfG [kWh]")

    """
    # Vygenerovanie obrazkov, kde sa zobrazuju vztahy medzi ufg a tokom
    if confObj.getLinearity():

        rel.visualRelations(confObj, decObj)

    # Vygenerovanie obrazkov, kde sa analyzuju data napr. z pohladu rocny, ci tyzdennych priemerov
    if confObj.getGroups():

        grp.visualGroups(confObj, decObj)

    # Analyza kazdej jednej nezavislej premennej, kazdej jednej rady, ktora vstupuje do modelu
    # 1. Statisticky opis rady
    # ds.descriptive(confObj, decObj)
    """

    # Spracovanie modelu. Polozka 0 v hore preddefinovanom zozname uloh.
    modelObj = prc.processModel(confObj, decObj)

    # Ak je pozadovane, aby bola vykonana zhodnotenie modelu pomocou metrik
    # Je potreba doplnit R2 a celkovo precistit metriky

    # Ak je pozadovane, tak vyrobit triedu, ktora sa povenuje stratam
    #  mapa strat
    #  analyza, ktora by povedala narast ci pokles dennecy stat k predposlednemu
    #  dnu alebo k medianu strat pre dany tok... ktora stanica v pomere k percentu
    #  straty aku ma stratu. zoznam stanic s najvyznamnejsimi stratami a tak podobne

    # ak je potreba, tak predikciu resp forecast, tak ako to je v stavajucom

    # Ak je pozadovana validacia modelu, tak model validujem.
    if confObj.getValidation():

        vl.validation(confObj, decObj, modelObj)

    """
    # ZAKOMENTOVANA CAST KODU FUNGUJE, ALE PRE TESTOVANIE VALIDACIE JU NEPOTREBUJEM.

    # vyhladenie realnych ufg dat a spocitanie relativnych ufg
    realSmtObj = smt.smoothSeries(modelObj.getRealUfG(), confObj.getSmoothingMethod(),
                                  confObj.getSmoothingBin(), confObj)

    # prepinac medzi alternativnym a regularnym modelom
    if confObj.getAlternativeModel():

        model = modelObj.getAltModel()
    else:

        model = modelObj.getModel()

    # Vyhladenie rady modelu pat dnovym priemerom a spocitanie relativnych UfG
    calcSmtObj = smt.smoothSeries(model, confObj.getSmoothingMethod(),
                                  confObj.getSmoothingBin(), confObj)

    # Porovnanie vyhladenych casovych radov
    cha.multilineChart(pd.DataFrame({"DATE": modelObj.getTimeVec(),
                                     "real": realSmtObj.getSmtSeries()[1:],
                                     "calc": calcSmtObj.getSmtSeries()[1:]}),
                       title="Porovnanie vyhladenych realnych a modelovanych strat",
                       xLabel="DATE", yLabel="Straty [kWh]")

    # Zobrazenie rozdielov dat v linecharte. Pomocou flatten, ndarray rozmer 2D prekonvertujem na 1D
    res = pd.Series(modelObj.getRealUfG().flatten()-model)

    cha.lineChart(decObj.getDF()["DATE"], res,
                  title="Zobrazenie rozdielu medzi modelom a realnymi hodnotami",
                  xLabel="Datum", yLabel="Sledovany rozdiel hodnot")

    # histogram rozdielov
    cha.histChart(res, title="Zobrazenie rozlozenia rozdielov")

    # QQ plot rozdielov
    cha.qqChart(res, "QQ plot zobrazujuci stav rozdielov z pohladu pripadneho normalneho rozdelenia")

    # Metriky vyhladenych dat
    # mt.metrics(np.array(realSmtObj.getSmtSeries()), np.array(calcSmtObj.getSmtSeries()))
    """
    """
    ###################################################################################################
    # Spocitanie relativnych strat
    ufgObj = ufg.processUfg(confObj.getAlternativeModel(), modelObj)

    # Porovnanie realtivnych [%] casovych radov
    r = ufgObj.getRealUfG()
    c = ufgObj.getCalcUfG()
    real = [float(r) for r in r]
    calc = [float(c) for c in c]
    cha.multilineChart(pd.DataFrame({"DATE": modelObj.getTimeVec(),
                                     "real": real,
                                     "calc": calc}),
                       title="Porovnanie relativnych realnych a modelovanych strat",
                       xLabel="DATE", yLabel="Straty [%/100]")

    # Metriky
    # mt.metrics(ufgObj.getRealUfG(), ufgObj.getCalcUfG())

    # vyhladenie realnych ufg dat a spocitanie relativnych ufg
    realRelSmtObj = smt.smoothSeries(ufgObj.getRealUfG(), confObj.getSmoothingMethod(),
                                     confObj.getSmoothingBin(), confObj)

    # Vyhladenie rady modelu pat dnovym priemerom a spocitanie relativnych UfG
    calcRelSmtObj = smt.smoothSeries(ufgObj.getCalcUfG(), confObj.getSmoothingMethod(),
                                     confObj.getSmoothingBin(), confObj)

    # Porovnanie vyhladenych casovych radov
    cha.multilineChart(pd.DataFrame({"DATE": modelObj.getTimeVec(),
                                     "real": realRelSmtObj.getSmtSeries()[1:],
                                     "calc": calcRelSmtObj.getSmtSeries()[1:]}),
                       title="Porovnanie vyhladenych relativnych realnych a modelovanych strat",
                       xLabel="DATE", yLabel="Straty [%/100]")

    # Metriky vyhladenych dat
    # mt.metrics(np.array(realSmtObj.getSmtSeries()), np.array(calcSmtObj.getSmtSeries()))
    """
    return decObj, confObj  # , modelObj


# Spustenie spracovania dat
if __name__ == "__main__":

    # data,  conf = run()
    # testSSA()
    decObj, data, regObj, pred, cv = testMLR()
