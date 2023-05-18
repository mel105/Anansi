#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 16:56:16 2023

@author: mel

Trieda vrati hodnoty modelu, tak aj potrebne prve derivacie.

"""

import numpy as np
import pandas as pd


class model:
    """
    Trieda ma za ulohu:
        1. vratit hodnoty modelu strat a prve derivacie tokov
    """

    def __init__(self, coef, inp, inter, calcDeriv=True, calcModel=True):
        """
        Konstruktor triedy model.

        Parameters
        ----------
        coef : TYPE
            DESCRIPTION. Vektor koeficientov
        inp : TYPE
            DESCRIPTION. Pole tokov
        inter : TYPE
              DESCRIPTION. True, do modelu chceme zapocitat aj absolutny koeficient. Casom asi odstranit.

        Returns
        -------
        None.

        """

        # odhad derivacii
        if calcDeriv:

            self._derivative(inp, inter)
        else:

            self._A = []

        # odhad modelu
        if calcModel:

            self._estimation(coef, inp, inter)
        else:

            self._val = []

    # get functions
    def losses(self):
        """
        Funkcia vrati odhad stat v podobe DataFrame. Potreba doplnit header a pripadne time column ako index.

        Returns
        -------
        None.

        """
        return pd.DataFrame(self._loss)

    def estimation(self):
        """
        Funkcia vrati odhadnute hodnoty modelu
        """

        return self._val

    def derivative(self):
        """
        Funkcia vrati maticu derivacii

        Returns
        -------
        TYPE
           DESCRIPTION.
        """

        return self._A

    def _derivative(self, inp, inter):
        """
        Funkcia vrati matici dizajnu

        Returns
        -------
        None.

        """

        if inter:
            # v pripade, ak by som chcel modelovat aj abs. clek. V jacobi matici, prvy stlpec jednickovy.
            self._A = np.c_[np.ones(len(inp)), inp]
        else:
            # inak
            self._A = inp

    def _estimation(self, coef, inp, inter):
        """
        Funkcia obsahuje model, ktory fitujem. Na vstupe sa nachadzaju jak odhadnute koeficienty, tak aj
        vstupne data tokov. Vysledkom je model strat, ktory ziskam prenasobenim koeficientov a tokov.

        Returns
        -------
        None.

        """

        # kontrola, ze ci coef je slpcovy alebo riadkovy vektor. Potrebujem stlpcovy
        msize = coef.shape
        if msize[0] == 1:
            coef = coef.T

        # prenasobenie koeficientov tokmi: mu + A*kA + B*kB + C*kC = model
        self._val = []
        self._loss = []

        if inter:

            for i in range(len(inp)):

                tmp = float(coef[0])
                arr = np.empty(len(coef), dtype=float)
                arr[0] = float(coef[0])

                for j in range(len(coef)-1):

                    ls = float(coef[j+1])*float(inp[i, j])
                    # print(i, j, coef[j+1], inp[i, j], ls)
                    # tmp += float(coef[j+1])*float(inp[i, j])
                    arr[j] = ls
                    tmp += ls

                self._val.append(tmp)
                self._loss.append(arr)
        else:

            for i in range(len(inp)):

                tmp = 0

                for j in range(len(coef)):

                    # print(" f.2%   f.2%", i, j)
                    tmp += coef[j]*float(inp[i, j])

                self._val.append(tmp)
