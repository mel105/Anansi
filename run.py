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

# from scipy.fft import fft, fftfreq, fftshift
# from scipy.signal import blackman


# import lib.plotitko as mplt
import numpy as np
import lib.config as cfg
import lib.processModel as prc
import lib.smoothSeries as smt
import lib.plotter as mplt
import lib.metrics as mt
import lib.processUfg as ufg
import lib.decoder as dc
import lib.descriptive as ds


def run():
    # Nacitanie koniguracneho suboru
    confObj = cfg.config()

    # Nacitanie dat
    fileName = confObj.getInpFileName()
    filePath = confObj.getInpFilePath()
    decObj = dc.decoder(fileName, filePath, confObj)

    # Analyza kazdej jednej nezavislej premennej, kazdej jednej rady, ktora vstupuje do modelu
    # 1. Statisticky opis rady
    ds.descriptive(confObj, decObj)

    # Spracovanie modelu. Polozka 0 v hore preddefinovanom zozname uloh.
    modelObj = prc.processModel(confObj, decObj)

    # Metriky nevyhladenych dat
    mt.metrics(modelObj.getRealUfG(), modelObj.getModel())

    """
    # vyhladenie realnych ufg dat a spocitanie relativnych ufg
    realSmtObj = smt.smoothSeries(modelObj.getRealUfG(), confObj.getSmoothingMethod(),
                                  confObj.getSmoothingBin(), confObj)

    # Vyhladenie rady modelu pat dnovym priemerom a spocitanie relativnych UfG
    calcSmtObj = smt.smoothSeries(modelObj.getModel(), confObj.getSmoothingMethod(),
                                  confObj.getSmoothingBin(), confObj)
    
    # Porovnanie vyhladenych casovych radov
    mplt.plotResults(modelObj.getTimeVec(), realSmtObj.getSmtSeries()[1:],
                     calcSmtObj.getSmtSeries()[1:])
    mplt.plotDetails(modelObj.getTimeVec(), np.array(realSmtObj.getSmtSeries()[1:]),
                     np.array(calcSmtObj.getSmtSeries()[1:]), confObj.getDBeg(), confObj.getDEnd())

    # Metriky vyhladenych dat
    mt.metrics(np.array(realSmtObj.getSmtSeries()), np.array(calcSmtObj.getSmtSeries()))

    ###################################################################################################
    # Spocitanie relativnych strat
    ufgObj = ufg.processUfg(modelObj)

    # Metriky nevyhladenych dat
    mt.metrics(ufgObj.getRealUfG(), ufgObj.getCalcUfG())

    # Porovnanie vyhladenych casovych radov
    mplt.plotResults(modelObj.getTimeVec(), ufgObj.getRealUfG(), ufgObj.getCalcUfG())
    mplt.plotDetails(modelObj.getTimeVec(), np.array(ufgObj.getRealUfG()),
                     np.array(ufgObj.getCalcUfG()), confObj.getDBeg(), confObj.getDEnd())

    # vyhladenie realnych ufg dat a spocitanie relativnych ufg
    realRelSmtObj = smt.smoothSeries(ufgObj.getRealUfG(), confObj.getSmoothingMethod(),
                                     confObj.getSmoothingBin(), confObj)

    # Vyhladenie rady modelu pat dnovym priemerom a spocitanie relativnych UfG
    calcRelSmtObj = smt.smoothSeries(ufgObj.getCalcUfG(), confObj.getSmoothingMethod(),
                                     confObj.getSmoothingBin(), confObj)

    # Porovnanie vyhladenych casovych radov
    mplt.plotResults(modelObj.getTimeVec(), realRelSmtObj.getSmtSeries()[1:],
                     calcRelSmtObj.getSmtSeries()[1:])
    mplt.plotDetails(modelObj.getTimeVec(), np.array(realRelSmtObj.getSmtSeries()[1:]),
                     np.array(calcRelSmtObj.getSmtSeries()[1:]), confObj.getDBeg(), confObj.getDEnd())

    # Metriky vyhladenych dat
    mt.metrics(np.array(realSmtObj.getSmtSeries()), np.array(calcSmtObj.getSmtSeries()))
    """


# Spustenie spracovania dat
if __name__ == "__main__":
    run()
