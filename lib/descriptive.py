#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:20:00 2023

@author: mel
"""

from tabulate import tabulate
import numpy as np
import scipy.stats as sc
from tabulate import tabulate


class descriptive:

    def __init__(self, confObj, decObj):
        """
        Konstruktor triedy descriptive, ktora ma za ulohu opisat data ulozene v kontajnere dataframe.
        Zrejme logickejsie by bolo, keby na vstupe bola konkretna rada (treba by mohla vzniknut nejaka
        funkcia), no vzhladom k tomu, ze riesim typ ulohy z viacrozmernymi datami, tak trieda sluzi k
        spracovaniu kompletneho dataframu


        Returns
        -------
        None.

        """

        self._stations = confObj.getDescStations()
        self._verb = confObj.getDescVerbosity()

        self._df = decObj.getDF()

        if ("DATE" in self._df.columns) | ("TIME" in self._df.columns):
            print("TIME OK")
            # MELTODO: mozno opis min time, max time, time resolution atp.

        # ak je zoznam stations =  All, potom prechadzam cez cely df, inak cez stations
        if self._stations[0] == "All":
            _, b = self._df.shape

            # slovnik, do ktoreho nastrkame statistiky  za kazdu sledovanu velicinu
            self._mdict = {}

            for i in range(1, b):
                stat = self._df.columns[i]
                vec = self._df[stat].to_numpy()

                res = self._statistics(vec, self._verb)

                # zoradena tabulka odhadnutych statistik
                tab = [["Mean", res[0]], ["Variance", res[1]], ["Sum", res[2]], ["Min", res[3]],
                       ["Q25", res[4]], ["Median", res[5]], ["Q75", res[6]], ["Max", res[7]],
                       ["Range", res[8]], ["Skewness", res[9]], ["Kurtosis", res[10]]]

                # naplnenie slovnika
                self._mdict[stat] = tab

                # pokial povolim, tak tabulka sa vypise na plochu
                if self._verb:
                    print("\n", "      ", stat, "\n", tabulate(
                        tab, headers=["Variable", "Estimation"]))

    def _statistics(self, vec, verb=0):
        """
        Funkcia vrati sledovane statistiky v zozname list

        Parameters
        ----------
        vec : TYPE
            DESCRIPTION.

        Returns
        -------
        res : TYPE
            DESCRIPTION.

        """

        res = []

        # mean
        mean = np.mean(vec)
        res.append(mean)
        # standard deviation
        # sdev = np.st(vec)
        # res.append(sdev)

        # variance
        var = np.var(vec)
        res.append(var)

        # sum
        suma = sum(vec)
        res.append(suma)

        # modal
        # mod = np.mod(vec)
        # res.append(mod)

        # min
        minv = np.quantile(vec, 0.00)
        res.append(minv)

        # quartile 25
        q25 = np.quantile(vec, 0.25)
        res.append(q25)

        # median
        q50 = np.quantile(vec, 0.50)
        res.append(q50)

        # quartile 75
        q75 = np.quantile(vec, 0.75)
        res.append(q75)

        # max
        maxv = np.quantile(vec, 1.00)
        res.append(maxv)

        # range
        rangev = q75 - q25
        res.append((rangev))

        # iqr

        # skew
        skewness = sc.skew(vec, axis=0, bias=True)
        res.append(skewness)

        # kurtosis
        kurt = sc.kurtosis(vec, axis=0, bias=True)
        res.append(kurt)

        return res
