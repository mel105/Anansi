#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:34:18 2023

@author: mel
"""

import numpy as np
import pandas as pd
import math as mt


class smoothSeries:
    """
    Trieda ma za ulohu:
        1. na zaklade zvolenej metody vyhladit vstupnu casovu radu
    """

    def __init__(self, tSeries, method="movingAverage", crit=5, config=None):
        """
        Konstruktor objektu smoothSeries

        Parameters
        ----------
        tSeries : TYPE dataframe alebo pole?
            DESCRIPTION. Vstupna rada, ktoru chcem vyhladit
        method : TYPE string
            DESCRIPTION. Metoda, ktorou chceme radu vyhladit. Default je nastavena metoda movingAverage
        crit : TYPE int
            DESCRIPTION. Kriterium, napr. sirka kosa, pomocou ktoreho vyhladzujeme radu
        config: TYPE config object
            DESCRIPTION. Je to ukazatel na objekt config. Pouzijem ho v pripade, ak mam v nastaveni
            viacej parametrov, ktore sa mi do vyhladzovacej metody hodia, ale nechcem ich na vstupe
            (napr. v inych metodach su zbytocne. Rozhodni, ze ci crit nebudem volat z config objektu)

        Returns
        -------
        None.

        """
        self._series = tSeries
        self._method = method
        self._crit = crit
        self._config = config

        if self._method == "movingAverage":
            self._processMovingAverage()
        else:
            try:
                self._method
            except TypeError as e:
                print(f"Required method is not implemented yet!: {e}")

    # Get methods
    def getSmtSeries(self):
        """
        Metoda vrati vyhladenu casovu radu

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self._smtSeries

    # Protected methods

    def _processMovingAverage(self):
        """
        Methoda vyhladi casovy rad na zaklade zvolenej metody a dodatocnych kriterii

        Returns
        -------
        None.

        """

        # kontrola _crit. V tejto metode by crit malo byt cislo nezaporne a asi vacsie ako 1, inak
        # vyhladenie nedava zmysel
        self._checkCrit()

        if self._config.getSmoothingAlt() == "center":
            # Pozadujem, aby vyhladena hodota bola v strede, to znamena, ze do priemeru vezmem x hodnot
            # z minulosti a x hodnot z buducnosti.
            # postup je taky, ze crit podelim dvoma a podiel zaokruhlim smerom dole a hore. Vstupne
            # pole dat doplnim o prislusyn pocet nul v zavtislosi na zaokruhleni a potom cez loop len
            # jednoducho spocitam priemery
            pod = self._crit/2
            f = mt.floor(pod)
            c = mt.ceil(pod)-1

            # rozsirenie rady
            for i in range(f):
                self._series = np.insert(self._series, 0, 0.0)

            for i in range(c):
                self._series = np.append(self._series, 0.0)

            # moving average s interpolovanym bodov v strede okna
            smtSeries = []

            beg = 0
            end = self._crit-1

            while end <= len(self._series):

                tmpavg = 0.0
                for i in self._series[beg:end]:
                    tmpavg += i

                smtSeries.append(tmpavg/self._crit)
                beg += 1
                end += 1

            self._smtSeries = smtSeries
            print()
        else:
            # hodnota je pocitana z x hodnot z minulosti
            print()

    def _checkCrit(self):
        """
        Metoda skontroluje kriterium sirky vyhladzovacieho okna

        Returns
        -------
        None.

        """

        if self._crit <= 1:
            self._crit = 2
