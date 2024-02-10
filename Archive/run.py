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

# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn import metrics
# from sklearn.linear_model import LinearRegression
# from statsmodels.tools.tools import add_constant
# import statsmodels.api as sm
import warnings
import lib.SSA as ssa
# import lib.supportSSA as spssa
import lib.support as sp
import lib.charts as cha
# import lib.validation as vl
import lib.decoder as dc
import lib.processModel as prc
import lib.config as cfg


# import seaborn as sns
# from statsmodels.stats.outliers_influence import variance_inflation_factor

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn import metrics
# from sklearn.linear_model import LinearRegression
# from sklearn import linear_model
# from sklearn.metrics import r2_score
# from sklearn.cluster import DBSCAN
# from sklearn.datasets import make_classification
# from sklearn.cluster import KMeans

# from numpy import unique
# from numpy import where
# from sklearn.datasets import make_classification
# from sklearn.cluster import Birch


import pandas as pd
# import numpy as np
import math
# import matplotlib.pyplot as plt
# import mpld3
# from matplotlib.backends.backend_webagg import (
#     FigureManagerWebAgg, new_figure_manager_given_figure)
# from matplotlib.figure import Figure


# import lib.smoothSeries as smt
# import lib.metrics as mt
# import lib.processUfg as ufg
# import lib.descriptive as ds
import lib.visualRelations as rel
import lib.visualGroups as grp

warnings.filterwarnings("ignore")


# We will use some methods from the sklearn module

# from sklearn import linear_model
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score


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

    # Vyhladenie TOTAL NB pomocou SSA algoritmu a zobrazenie vysledkov
    if confObj.getSmoothingMethod() == "SSA":

        L = math.floor(decObj.getDF().shape[0]/2)  # max hodnota okna, v ktorom sa vysetruju vlastne vektory
        F_ssa = ssa.SSA(decObj.getDF()["TOTAL NB"], L)

        # Ulozenenie povodneho UfG do novej premennej TOTAL NB ORIG
        decObj.getDF()["TOTAL NB ORIG"] = decObj.getDF()["TOTAL NB"].copy()

        # Povodne UfG je nahradene vyhladenym UfG.
        S = 200  # vezmem si prvych S vlastnych vektorov, pomocou ktrorych zrekonstruujem orig. casovu radu
        decObj.getDF()["TOTAL NB"] = F_ssa.reconstruct(slice(0, S))

        print("\nWARNING: TOTAL NB was reconstructed by SSA algorithm. Please be carefull for \
              futher TOTAL NB parameter analysing\n")

        # Zobrazenie orig strat a vyhladenych strat, ak SSA je v nastaveni zvolena ako smoothed method.
        if confObj.getPlotSmoothingResults():

            # pokracovanie rekonstrukcie
            noise = F_ssa.reconstruct(slice(S+1, L))
            origs = F_ssa.orig_TS
            t = decObj.getDF()["DATE"]
            cha.ssaPlot(t, origs, decObj.getDF()["TOTAL NB"], noise)

            # kreslenie w-corr mapy
            F_ssa.plot_wcorr("AAAA")

        else:
            print("No SSA result plots!")
            # Nebude sa nic kreslit

    else:
        # tu by sa mohli nacitat postupne ine metody, napriklad tie zalozene na moving average, alebo Kernel
        # smoothing.
        print("Other Smoothing methods are not implemented yet!")

    # Vygenerovanie obrazkov, kde sa zobrazuju vztahy medzi ufg a tokom
    if confObj.getLinearity():

        rel.visualRelations(confObj, decObj)

    # Vygenerovanie obrazkov, kde sa analyzuju data napr. z pohladu rocny, ci tyzdennych priemerov
    if confObj.getGroups():

        grp.visualGroups(confObj, decObj)

    # Analyza kazdej jednej nezavislej premennej, kazdej jednej rady, ktora vstupuje do modelu
    # 1. Statisticky opis rady
    # ds.descriptive(confObj, decObj)

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
    # if confObj.getValidation():

    #     vl.validation(confObj, decObj, modelObj)

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

    data,  conf = run()
