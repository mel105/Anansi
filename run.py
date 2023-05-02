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

# import pandas as pd
import lib.config as cfg
import lib.processModel as prc
# import lib.smoothSeries as smt
# import lib.metrics as mt
# import lib.processUfg as ufg
import lib.decoder as dc
import lib.validation as vl
# import lib.descriptive as ds
# import lib.charts as cha


def run():
    # Nacitanie koniguracneho suboru
    confObj = cfg.config()

    # Nacitanie dat
    fileName = confObj.getInpFileName()
    filePath = confObj.getInpFilePath()
    decObj = dc.decoder(fileName, filePath, confObj)

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
    if confObj.getValidation():

        vl.validation(confObj, decObj, modelObj)

    # ZAKOMENTOVANA CAST KODU FUNGUJE, ALE PRE TESTOVANIE VALIDACIE JU NEPOTREBUJEM.
    """
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

    # PO PREROBENI, DO HISTOGRAMU IDU NDARRAYS(1d)
    # histogram rozdielov
    cha.histChart(res, title="Zobrazenie rozlozenia rozdielov")

    # TU TO MUSIM ESTE OVERIT
    # QQ plot rozdielov
    cha.qqChart(res, "QQ plot zobrazujuci stav rozdielov z pohladu pripadneho normalneho rozdelenia")

    # Metriky vyhladenych dat
    # mt.metrics(np.array(realSmtObj.getSmtSeries()), np.array(calcSmtObj.getSmtSeries()))

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
    return decObj, modelObj, confObj


# Spustenie spracovania dat
if __name__ == "__main__":
    data, model, config = run()
