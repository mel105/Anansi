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

        self._df = pd.read_excel(filePath+"/"+fileName[0]+".xlsx")
        self._df.rename(columns={self._df.columns[0]: "DATE"}, inplace=True)

        # Filtrovanie dat podla zadanych limitnych datumov
        self._df = self._df[(self._df['DATE'] >= confObj.getBeg()) &
                            (self._df['DATE'] <= confObj.getEnd())]

    def getDF(self):
        """
        Funkcia vrati naplneny kontajner formatu Dataframe

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._df
