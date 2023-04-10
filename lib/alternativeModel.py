#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 22:36:33 2023

@author: mel
"""

import lib.model as md


class alternativeModel:

    def __init__(self, coef, inp, inter, maxLoss):
        """
        Konstruktor triedy alternativeModel.

        Parameters
        ----------
        coef : TYPE
            DESCRIPTION. Vektor koeficientov, ktore vstupuju do odhadu modelu
        inp : TYPE
            DESCRIPTION. Pole vstupnych dat
        inter : TYPE
            DESCRIPTION. Zapocitat alebo nezapocitat konstantny koeficient
        maxLoss : TYPE
            DESCRIPTION. Hodnota maximalnej straty. Koeficienty by nemali byt vacsie ako tato hodnota

        Returns
        -------
        None.

        """

        # dojde k prefiltrovaniu koeficientov. Tie koeficienty, ktore su vacsia ako max strata, tak tie
        # nahradime prave za maxLoss parameter
        self._filterCoefs(coef.T, maxLoss)

        # Objekt model okderm derivacie mdoelu obsahuje aj odhad modelu samotneho.
        modelObj = md.model(self._altCoef, inp, inter, calcDeriv=False, calcModel=True)
        self._valAlt = modelObj.estimation()

    # get funkcie
    def getAltModel(self):
        """
        Funkcia vrati hodnoty alternativneho modelu

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self._valAlt

    def getAltCoef(self):
        """
        Metoda vrati adu alternativnych koeficientov

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self._altCoef

    # clenske metody
    def _filterCoefs(self, coef, maxLoss):
        """
        Metoda sa postara o prefiltrovanie koeficientov.

        Parameters
        ----------
        coef : TYPE
            DESCRIPTION.
        maxLoss : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        altCoef = coef

        overLossIndiciesPlus = [n+1 for n, i in enumerate(coef[1:]) if i >= maxLoss]

        for i in overLossIndiciesPlus:
            altCoef[i] = maxLoss

        overLossIndiciesMinus = [n+1 for n, i in enumerate(coef[1:]) if i <= -maxLoss]

        for i in overLossIndiciesMinus:
            altCoef[i] = -maxLoss

        self._altCoef = altCoef
