#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:02:49 2023

@author: mel
"""

import pandas as pd
import numpy as np


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

        # uprava policka v dataframe, ak neobsahuje ziadnu hodnotu
        # dataframe plny true or false
        boo = self._df.applymap(np.isreal)

        # konvert boo do pola array
        boa = boo.to_numpy()

        # indexy/suradnice kde v boo najdem false
        ri, ci = np.where(~boa)
        for i in range(len(ri)):
            self._df.loc[ri[i], self._df.columns[ci[i]]] = 0.0

        self._dfred = self._df

        # kopia df matice. To pre neskorsiu validaciu
        self._dfFull = self._df

        # definovanie si stlpcov, ktore nie su numerickeho typu
        for i in self._dfred.columns:
            j = pd.to_numeric(self._dfred[i], errors='coerce').notnull().all()

            if j is False:
                self._dfred = self._dfred.drop(columns=i, axis=1)

        # Filtrovanie dat podla zadanych limitnych datumov
        self._df = self._df[(self._df['DATE'] >= confObj.getBeg()) &
                            (self._df['DATE'] <= confObj.getEnd())]

        self._dfred = self._dfred[(self._dfred['DATE'] >= confObj.getBeg()) &
                                  (self._dfred['DATE'] <= confObj.getEnd())]

        # Na obrazovku vytlac info o dataframe
        self._dfFull.info()

        # Rozsirenie full dataframe o styri stlpce: Year-Month, Year, Month, Week
        self._dfExt = self._df  # kopia full matice, ktoru rozsirim o nove polozky
        self._dfExt["YM"] = [i.strftime("%Y-%m") for i in list(self._df.DATE)]
        self._dfExt["YW"] = [i.strftime("%Y-%V") for i in list(self._df.DATE)]
        self._dfExt["YEAR"] = [i.strftime("%Y") for i in list(self._df.DATE)]
        self._dfExt["MONTH"] = [i.strftime("%m") for i in list(self._df.DATE)]
        self._dfExt["WEEK"] = [i.strftime("%V") for i in list(self._df.DATE)]

        # Grupovanie dat. Zmyslom je pripravit data zgrupovane posla roku, potom podla mesiaca a podla tyzdna.
        # Vysledkom su nove dataframes, s mesacnymi, rocnymi a tyzdennymi priemermi

        # Rocne priemery
        self._dfYearly = self._dfExt.groupby(["YEAR"]).mean()
        self._dfYearly = self._dfYearly.reset_index()

        # Mesacne priemery
        self._dfMonthly = self._dfExt.groupby(["YM", "MONTH"]).mean()
        self._dfMonthly = self._dfMonthly.reset_index()

        # Tyzdenne priemery
        self._dfWeekly = self._dfExt.groupby(["YW", "WEEK"]).mean()
        self._dfWeekly = self._dfWeekly.reset_index()

    def getDFYearly(self):
        """
        Vrati data s napocitanymi rocnymi priemermi

        Returns
        -------
        None.

        """
        return self._dfYearly

    def getDFMonthly(self):
        """
        Funkcia vrati dataframe s napocitanymi mesacnymi priemermi
        """

        return self._dfMonthly

    def getDFWeekly(self):
        """
        Funkcia vrati dataframe s napocitanymi tyzdennymi priemermi
        """

        return self._dfWeekly

    def getDFExt(self):
        """
        Funkcia vrati rozsirenu maticu o info YEAR, MONTH WEEK.

        Returns
        -------
        None.

        """

        return self._dfExt

    def getDFfull(self):
        """
        Funkcia vrati kompletne data, nie fitrovane od-do.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self._dfFull

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
