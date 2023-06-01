#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 22:02:07 2023

@author: mel
"""
import pandas as pd
import numpy as np

import lib.metrics as mt
import lib.support as sp
import lib.model as md
import lib.charts as ch


class validation:

    def __init__(self, config, decoder, model):
        """
        Konstruktor triedy. Moderuje postup validacie modelu.

        Parameters
        ----------
        config : TYPE
            DESCRIPTION. Objekt triedy config. Obsahuje settings
        decoder : TYPE
            DESCRIPTION. Objekt triedy decoder. Obsahuje nacitane data
        model : TYPE
            DESCRIPTION. Objekt triedy processModel. Obsahuje konstanty potrebne pre overenie presnosti modelu

        Returns
        -------
        None.

        """

        if config.getPrediction():
            # Predikcia:

            # pripravim si dataframe dat, na ktorych chcem predikovat.
            d = decoder.getDFfull()
            lastTime = str(d.DATE.iat[-1])
            setEndTime = config.getEnd()

            # pokial setEndTime > posledna hodnota v DF, potom nema zmysel robit nejaku predikciu
            if setEndTime >= lastTime:

                print("No Validation")
            else:

                # priprav data: od povodneho DF si vezmem data od setEndTime -> End.
                de = d[(d['DATE'] > config.getEnd()) & (d['DATE'] <= d['DATE'].iat[-1])]
                # print(de)

                # priprava dat do validacie
                data = sp.fillDataContainer(de, config.getOutStations())

                # funkcia, ktora prevezme data a koeficienty a spocita UFG
                modelObj = md.model(model.getModelRegressors(), data, config.getAddIntercept(),
                                    calcDeriv=False, calcModel=True)

                pred = modelObj.estimation()

                # pripravim si realne UFG
                if config.getSmoothingMethod() == "SSA":
                    real = de["TOTAL NB ORIG"]
                else:
                    real = de["TOTAL NB"]

                # data na vystup
                dfvalid = pd.DataFrame()
                dfvalid["DATE"] = de["DATE"]
                dfvalid["PRED"] = pred
                dfvalid["REAL"] = real

                # spocitam metriky
                rmse, _, _, _ = mt.metrics(dfvalid.REAL.to_numpy(), dfvalid.PRED.to_numpy(), 1)

        else:
            print(config.gerPrediction())
            # Cross-validation:

            # funkcia ktora vyberie nejake percento nahodnych dat a vyrobi z nich dataframe.

            # funkcia, ktora prevezme data a koeficienty a spocita UFG

            # pripravim si realne UFG

            # spocitam metriky

        if config.getValidchart():

            # zavolaj funkcie na vykreslenie predikcie

            # MODEL
            mtim = model.getTimeVec()
            mdat = model.getModel()

            # PREDIKCE
            ptim = de["DATE"]
            ci = 1.96 * np.asarray(rmse)

            # PLOT
            ch.prediction(mtim, mdat, ptim, real, np.array(pred), ci)
