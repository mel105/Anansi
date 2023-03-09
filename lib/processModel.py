#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:53:09 2023

@author: mel
"""

import pandas as pd

import lib.lsq as lsq
import lib.support as msupp
import lib.decoder as dec
import lib.plotter as mplt


class processModel:
    """
    Trieda ma za ulohu:
        1. spracovat data tokov (zaistenie nacitania dat - samotne nacitanie v decoder)
        2. navrhnut model calc UfG (pomocou linearneho modelu a nasledneho fitovania pomocou LSQ)
        3. sumarizovat kvalitu spracovania (model summary: okrem statistik, histogram reziduii, R2 atp)
    """

    def __init__(self, confObj):
        """
        Konstruktor tiedy processModel

        Parameters
        ----------
        confObj : TYPE config
            DESCRIPTION. objekt triedy config

        Returns
        -------
        None.

        """

        self._conf = confObj

        # Zoznam stanic, s ktorymi budem pracovat
        self._listOfStations = self._conf.getOutStations()

        # Nacitanie excelovskej tabulky
        self._df = dec.decoder(self._conf.getInpFileName(), self._conf.getInpFilePath(), self._conf)

        # Spracovanie vybranych dat a ich ulozenie do pola data
        self._data = msupp.fillDataContainer(self._df, self._listOfStations)

        # Nastavenie vah
        self._weightVec = msupp.calcWeights(confObj.getCalcWeights, self._df)

        # Vektor merani
        self._lVec = pd.DataFrame().assign(total=self._df["TOTAL NB"]).to_numpy()

        # Pravdepodobnost, na ktorej pocitam 100P% kvantil rozdelenia
        self._probUp = self._conf.getProbUp()

        # Spracovanie modelu
        self._dataProcessing()

        # Statistiky kvality vyhladenia modelu: histogram rezisudii R2 koef atd atp.
        # self._summary()

        # obrazky
        # Vykreslenie rozdielu vstupneho toku plynu a vystupneho toku plynu
        mplt.plotResults(self._df["DATE"], self._lVec, self._valEst)

        # Zobrazenie detailu
        mplt.plotDetails(self._df["DATE"], self._lVec, self._valEst, self._conf.getDBeg(),
                         self._conf.getDEnd())

    # GET FUNKCIE
    def getModel(self):
        """
        Metoda vrati odhadnute modelove hodnoty

        Returns
        -------
        None.

        """

        return self._valEst

    def getRealUfG(self):
        """
        Metoda vrati realne hodnoty UfG

        Returns
        -------
        None.

        """

        return self._lVec

    def getIntakes(self):
        """
        Metoda vrati hodnoty intakes tokov.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._df["INTAKES"]

    def getTimeVec(self):
        """
        Metoda vrati rcasovy vektor

        Returns
        -------
        None.

        """

        return self._df["DATE"]

    # PROTECTED FUNKCIE
    def _dataProcessing(self):
        """
        Metoda moderuje spracovanie dat.

        Returns
        -------
        None.

        """

        stopDataCrit = True
        calcFitStat = False

        numIterData = 0

        # Algoritmus vypoctu. Testujem koeficient v cykle vyhadzujem tie, ktore mi nevyhovuju podmienke
        while stopDataCrit:

            numIterData = numIterData + 1
            print(f"\n * Iteracia (DATA) c. {numIterData}")

            # Rozmer datoveho kontajnera
            [nrows, ncols] = self._data.shape

            # Algoritmus LSQ vyrovnania
            coef, A, dh, Qvv, N, valEst = lsq.processLSQ(self._data, self._lVec, self._weightVec,
                                                         calcFitStat, self._listOfStations, self._conf)

            # Tu nasleduje zhodnotenie koeficientov a vylucenie toho signalu, ktoreho koeficient je
            # najvacsi a nesplnuje podmienku, aby jeho hodnota bola  Coef >= -0.01 && Coef <=0.01
            stopDataCrit, coefOut, dataOut, maxIdx, maxElem, listOfStationsOut = msupp.reduceDataArray(
                coef, self._data, self._listOfStations, self._conf)

            # Update povodnych kontajnerov
            self._data = dataOut
            coef = coefOut
            self._listOfStations = listOfStationsOut

        # Celkove calc UfG
        self._valEst = valEst

        # Statistiky po vyrovnani modelu
        lsq.summary(self._lVec, self._valEst, self._listOfStations, A, dh,  Qvv, N, coef, self._probUp)
