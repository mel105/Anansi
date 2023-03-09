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
# import lib.ufg as ufg


def run():
    # Nacitanie koniguracneho suboru
    confObj = cfg.config()

    # Analyza kazdej jednej nezavislej premennej, kazdej jednej rady, ktora vstupuje do modelu

    # Spracovanie modelu. Polozka 0 v hore preddefinovanom zozname uloh.
    modelObj = prc.processModel(confObj)

    # vyhladenie realnych ufg dat a spocitanie relativnych ufg
    realSmtObj = smt.smoothSeries(modelObj.getRealUfG(), confObj.getSmoothingMethod(),
                                  confObj.getSmoothingBin(), confObj)

    # Vyhladenie rady modelu pat dnovym priemerom a spocitanie relativnych UfG
    calcSmtObj = smt.smoothSeries(modelObj.getModel(), confObj.getSmoothingMethod(),
                                  confObj.getSmoothingBin(), confObj)
    # Metriky nevyhladenych dat
    mt.metrics(modelObj.getRealUfG(), modelObj.getModel())

    # Porovnanie vyhladenych casovych radov
    mplt.plotResults(modelObj.getTimeVec(), realSmtObj.getSmtSeries()[1:],
                     calcSmtObj.getSmtSeries()[1:])
    mplt.plotDetails(modelObj.getTimeVec(), np.array(realSmtObj.getSmtSeries()[1:]),
                     np.array(calcSmtObj.getSmtSeries()[1:]), confObj.getDBeg(), confObj.getDEnd())

    # Metriky evyhladenych dat
    mt.metrics(np.array(realSmtObj.getSmtSeries()), np.array(calcSmtObj.getSmtSeries()))

    # Spocitanie relativnych strat
    # ufgObj = ufg.processUfg(confObj, modelObj)


# Spustenie spracovania dat
if __name__ == "__main__":
    run()
