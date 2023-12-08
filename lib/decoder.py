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

        # nacitanie originalnych dat, ktore si pripravime v excel programe
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

        # kopia df matice. To pre neskorsiu validaciu pripradne zalohu povodnych dat, pretoze v df neskor
        # data centrujem o priemer.
        self._dfFull = self._df.copy()

        # Na obrazovku vytlac info o dataframe. dfFull teda obsahuje uplne orig data
        self._dfFull.info()

        # Centrovanie dat
        # t.j. od kazdeho stlpca odcitam jeho priemernu hodnotu. Malo by to pomoct potlacit kolinearitu medzi
        # datmi. Najprv odstranim stlpec DATE a mozno aj TOTAL NB, data zcentrujem a potom odstranene stlpce
        # opat pridam do dataframeu
        if confObj.getCentering():

            tm = self._df["DATE"]
            self._df = self._df.apply(lambda x: x-x.mean())
            self._df["DATE"] = tm

        # Linearna zavislost
        # Vysetrenie linearnej zavislosti vektorov vo vstupnych datach a to z dovodu identifikacie, ktore
        # stlpce dat su ako zavisle
        if confObj.getInvestLin():

            matrix = self._df
            matrix = matrix.drop(columns="DATE")
            matrix = matrix.to_numpy()
            ld = []

            for i in range(matrix.shape[1]):

                for j in range(matrix.shape[1]):

                    if i != j:
                        inner_product = np.inner(
                            matrix[:, i],
                            matrix[:, j]
                        )
                        norm_i = np.linalg.norm(matrix[:, i])
                        norm_j = np.linalg.norm(matrix[:, j])

                        if np.abs(inner_product - norm_j * norm_i) < 1E-1:
                            # print(i, j, 'Dependent')
                            ld.append(i)

            # print(ld)

        # Orezanie df casovej rady podla beg, end definovane v config
        # dfred je copy originalne df s typ, ze ale tam bude neskor orezana podla toho, ako mam v configu
        # nastaveny beg, end.
        self._dfred = self._df.copy()

        # definovanie si stlpcov, ktore nie su numerickeho typu
        for i in self._dfred.columns:
            j = pd.to_numeric(self._dfred[i], errors='coerce').notnull().all()

            if j is False:
                self._dfred = self._dfred.drop(columns=i, axis=1)

        self._dfred = self._dfred[(self._dfred['DATE'] >= confObj.getBeg()) &
                                  (self._dfred['DATE'] <= confObj.getEnd())]

        # VYROBENIE PRIEMEROV
        # Rozsirenie full dataframe o styri stlpce: Year-Month, Year, Month, Week. Pre tento ucel pouzival
        # dalsiu copy df.
        self._dfExt = self._df.copy()  # kopia full matice, ktoru rozsirim o nove polozky
        self._dfExt["YM"] = [i.strftime("%Y-%m") for i in list(self._df.DATE)]
        self._dfExt["YW"] = [i.strftime("%Y-%V") for i in list(self._df.DATE)]
        self._dfExt["YEAR"] = [i.strftime("%Y") for i in list(self._df.DATE)]
        self._dfExt["MONTH"] = [i.strftime("%m") for i in list(self._df.DATE)]
        self._dfExt["WEEK"] = [i.strftime("%V") for i in list(self._df.DATE)]

        # Grupovanie dat. Zmyslom je pripravit data zgrupovane podla roku, potom podla mesiaca a podla tyzdna.
        # Vysledkom su nove dataframes, s mesacnymi, rocnymi a tyzdennymi priemermi

        # Rocne priemery
        self._dfYearly = self._dfExt.groupby(["YEAR"]).mean(numeric_only=True)
        self._dfYearly = self._dfYearly.reset_index()

        # Mesacne priemery
        self._dfMonthly = self._dfExt.groupby(["YM", "MONTH"]).mean(numeric_only=True)
        self._dfMonthly = self._dfMonthly.reset_index()

        # Tyzdenne priemery
        self._dfWeekly = self._dfExt.groupby(["YW", "WEEK"]).mean(numeric_only=True)
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
        Funkcia vrati rozsirenu maticu o info YEAR, MONTH WEEK. Ak je v config subore pozadovane, aby sme data
        redukovali o priemery, tak data su centrovane.

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
        heat matp diagramu alebo scatter mastrix diagramu. Navyse je matica casovo orezana o beg, end,
        definovane v configure subore a ak je v config este pozadovane centrovanie na nulu, tak red. je tiez
        centrovana.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._dfred

    def getDF(self):
        """
        Funkcia vrati naplneny kontajner formatu Dataframe. Matica je redukovana o priemery, teda centrovana
        na nulu, ak to v configuraku pozadujem. Inak je to podobne ako dffull orig zdroj dat.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._df
