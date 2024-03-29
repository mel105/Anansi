#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:53:09 2023

@author: mel
"""

import pandas as pd
import numpy as np

import lib.lsq as lsq
import lib.support as msupp
# import lib.decoder as dec
# import lib.plotter as mplt
import lib.alternativeModel as alm


class processModel:
    """
    Trieda ma za ulohu:
        1. spracovat data tokov (zaistenie nacitania dat - samotne nacitanie v decoder)
        2. navrhnut model calc UfG (pomocou linearneho modelu a nasledneho fitovania pomocou LSQ)
        3. sumarizovat kvalitu spracovania (model summary: okrem statistik, histogram reziduii, R2 atp)
    """

    def __init__(self, confObj, decObj):
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
        self._df = decObj.getDF()
        # self._df = dec.decoder(self._conf.getInpFileName(), self._conf.getInpFilePath(), self._conf)

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
        # mplt.plotResults(self._df["DATE"], self._lVec, self._valEst)

        # Zobrazenie detailu
        # mplt.plotDetails(self._df["DATE"], self._lVec, self._valEst, self._conf.getDBeg(),
        #                  self._conf.getDEnd())

    # GET FUNKCIE
    def getModelRegressors(self):
        """
        Metoda vrati regresory, cize koeficienty zvoleneho modelu

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._coef

    def getAltModelRegressors(self):
        """
        Metoda vrati regresory, cize koeficienty zvoleneho alternativneho modelu

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._coefAlt

    def getModel(self):
        """
        Metoda vrati odhadnute modelove hodnoty

        Returns
        -------
        None.

        """

        return self._valEst

    def getAltModel(self):
        """
        Metoda vrati alternativny model
        """

        return self._valAlt

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

        return self._df["Intakes"].to_numpy()

    def getAnalysedData(self):
        """
        Funkcia vrati data, s ktorymi pracujem napr. v casti lsq.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self._data

    def getListOfAnalysedStations(self):

        return self._listOfStations

    def getTimeVec(self):
        """
        Metoda vrati rcasovy vektor

        Returns
        -------
        None.

        """

        return self._df["DATE"]

    def getLosses(self):
        """
        Metoda vrati straty rozpocitane po case a staniciam

        Returns
        -------
        None.

        """
        return self._loss

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
            coef, A, dh, Qvv, N, valEst, losses = lsq.processLSQ(self._data, self._lVec, self._weightVec,
                                                                 calcFitStat, self._listOfStations, self._conf)

            """
            # K uvahe. Tu by mohla nasledovat procedura, v ktorej prebehne:
            # A.) zhodnotenie koeficientov a vylucenie toho signalu, ktoreho koeficient je najvacsi a
            # nesplnuje podmienku, aby jeho hodnota bola  Coef >= -0.01 && Coef <=0.01. Tu dochadza k postupu
            # podla ktoreho po vyluceni toku z vysokym koeficientom odznova prepocitavam fitovanie a procest
            # vylucovania tokov a noveho fitovania konci vtedy, ked vsetky koeficienty su pod preferovanou
            # hodnotou. Proces je zdlhavy a moze odfiltrovat velka tokov. Rozumnejsie by bolo rucne nastavovat
            # konfiguraciu a trafit sa do takej sustavy, ktora bude z pohladu vyrovnania rozumna.
            stopDataCrit, coefOut, dataOut, maxIdx, maxElem, listOfStationsOut = msupp.reduceDataArray(
                coef, self._data, self._listOfStations, self._conf)

            # Update povodnych kontajnerov
            self._data = dataOut
            coef = coefOut
            self._coef = coef
            self._listOfStations = listOfStationsOut

            """

            self._coef = coef

            if self._conf.getAddIntercept():

                self._listOfStations.insert(0, "Constant")

            stopDataCrit = False
            # B.) prenastavim vsetky koeficienty, ktore su vacsie ako pozadovany limitna hodnota koeficientov
            # prave na hodnotu limit koeficientov. To urobim mimo While loop. Problem je ale ten, ze k nim
            # nebudeme mat Qvv ani ziadne ine statistiky, ktore by mam poukazali na presnost koeficientov.
            # alebo vyrobim este nieco, co nazveme alternative model. Vysledkom budu filtrovane koef a valEst.

        # Celkove calc UfG
        self._valEst = valEst

        # Rozpocitane straty podla koeficientov a tokov v kazdom case
        header = self._conf.getOutStations()

        # if "Constant" in header:
        #    header.pop(0)

        losses.columns = header
        self._loss = losses

        # Save the losses
        outPath = self._conf.getOutLocalPath()+"/"+self._conf.getCsvFolderName()
        np.savetxt(outPath + "/" + "losses.csv",
                   self._loss,
                   delimiter=", ",
                   header=",".join(header),
                   newline="\n",
                   comments="",
                   fmt="%f")

        print(self._loss)

        # Statistiky po vyrovnani modelu
        lsq.summary(self._lVec, self._valEst, self._listOfStations, A, dh,  Qvv, N, coef.T, self._probUp,
                    self._conf)

        # Alternative model
        if self._conf.getAlternativeModel():

            altObj = alm.alternativeModel(coef, self._data, self._conf.getAddIntercept(),
                                          self._conf.getLimitRelativeUfG())

            self._valAlt = altObj.getAltModel()
            self._coefAlt = altObj.getAltCoef()
        else:

            self._valAlt = []
            self._coefAlt = []
