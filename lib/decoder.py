#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:02:49 2023

@author: mel
"""

import pandas as pd


class decoder:

    def __init__(self, fileName, filePath, confObj):
        """

        Parameters
        ----------
        fileName : TYPE
            DESCRIPTION.
            filePath : TYPE
            DESCRIPTION.
            confObj : TYPE
            DESCRIPTION.

            Returns
            -------
            df : TYPE
            DESCRIPTION.

            """

        self._df = pd.read_excel(filePath+"/"+fileName[0]+".xlsx", verbose=True)
        self._df.rename(columns={self._df.columns[0]: "DATE"}, inplace=True)
        self._dfred = self._df
        # definovanie si stlpcov, ktore nie su numerickeho typu
        for i in self._dfred.columns:
            j = pd.to_numeric(self._dfred[i], errors='coerce').notnull().all()

            if j == False:
                self._dfred = self._dfred.drop(columns=i, axis=1)

        # Filtrovanie dat podla zadanych limitnych datumov
        self._df = self._df[(self._df['DATE'] >= confObj.getBeg()) &
                            (self._df['DATE'] <= confObj.getEnd())]

        self._dfred = self._dfred[(self._dfred['DATE'] >= confObj.getBeg()) &
                                  (self._dfred['DATE'] <= confObj.getEnd())]

    def getDFred(self):
        """
        Funkcia vrati naplneny kontajner formatu Dataframe. Je ale redukovany o stlpce, ktore boli
        identifikovane ako take, ktore obsahuju nenumericke hodnoty. Tuto maticu pouzijem napr. v pripade
        heat matp diagramu alebo scatter mastrix diagramu.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._dfred

    def getDF(self):
        """
        Funkcia vrati naplneny kontajner formatu Dataframe

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._df
