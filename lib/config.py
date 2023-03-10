#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:31:10 2023

@author: mel
"""

import json


class config:
    """
    Trieda ma za ciel spracovat konfiguracny subor
    """

    def __init__(self):
        """
        Konstruktor
        Returns
        -------
        None.
        """

        self._loadConfig()

    # get funkcie
    def getInpFileName(self):
        """
        Funkcia vrati nazov pozadovaneho vstupneho suboru
        Returns
        -------
        TYPE string
            DESCRIPTION. Nazov pozadovaneho suboru
        """
        return self._inpFileName

    def getInpFilePath(self):
        """
        Funkcia vrati lokalnu cast cesty k csv datam
        Returns
        -------
        TYPE string
            DESCRIPTION. Lokalna cast cesty k csv data,
        """
        return self._inpLocalPath

    def getBeg(self):
        """
        Funkcia vrati pozadovany zaciatok spracovanej casovej rady

        Returns
        -------
        None.

        """

        return self._beg

    def getEnd(self):
        """
        Funkcia vrati pozadovany koniec spracovanej casovej rady

        Returns
        -------
        None.

        """

        return self._end

    def getDBeg(self):
        """
        Funkcia vrati pozadovany zaciatok spracovanej casovej rady ale len pre zobrazenie detailu

        Returns
        -------
        None.

        """

        return self._dbeg

    def getDEnd(self):
        """
        Funkcia vrati pozadovany koniec spracovanej casovej rady, ale len pre zobrazenie detailu rady

        Returns
        -------
        None.

        """

        return self._dend

    def getEpsVal(self):
        """
        Funkcia vrati presnost, pod ktoru by mala spradnut norma oprav v procese spracovania LSQ, aby
        sa vyrovnavanie modelu ukoncilo

        Returns
        -------
        None.

        """

        return self._epsVal

    def getMaxIter(self):
        """
        Funkcia vrati maximalny pocet iteracii, ktore sa pretocia v spracovani vyrovnania modelu
        pomocou LSQ metody. V pripade dopre inicializovaneho modelu nam postacia dve iter8cie. Viace
        iteracii signalizuje na problem bud s modelom alebo v datach.

        Returns
        -------
        None.

        """

        return self._maxIter

    def getLimitRelativeUfG(self):
        """
        Funkcia vrati hodnotu, ktora symbolizuje limitnu predpokladanu chybu v realnom UfG

        Returns
        -------
        None.

        """

        return self._limitRelativeUfG

    def getCalcWeights(self):
        """
        Funkcia vrati parameter typu boolear, ktory rozhodne, ze ci sa v casti spracovania modelu
        pouziju vahy (True) alebo matica vah bude jednotkova (False)

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self._calcWeights

    def getAddIntercept(self):
        """
        Funkcia vrati priznak, podla ktoreho sa rozhodnem, ze ci do modelu chcem absolutny clen (True)
        alebo nechcem (False)

        Returns
        -------
        _addIntercept : TYPE
            DESCRIPTION.

        """
        return self._addIntercept

    def getProbUp(self):
        """
        Funkcia vrati pravdepodobnost, na ktorej pocita 100P% kvantil rozdelenia

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self._probUp

    def getOutStations(self):
        """
         Funkcia vrati zoznam stanic, s ktorymi budem pracovat
         Returns
         -------
         TYPE pole
             DESCRIPTION.
         """
        return self._listStations

    def getSmoothingMethod(self):
        """
        Metoda vrati volbu vyhladzovacej metody. Casom by mohli byt implmentovane tieto postupy:
            * Moving Average
            * Moving Median
            * Kernel Smoother
            * Kalman Smoother

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self._smoothingMethod

    def getSmoothingBin(self):
        """
        Metoda vrati nejake vyhladzovacie kriterium. Napr. v pripade Moving average metody to zrejme
        bude pocet jednotiek, z ktorych spocitam priemer. V pripade Kernel smoother treba hodnotu
        gaussovskeho jadra a tak podobne.
        return self._smoothingBin

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._smoothingBin

    def getSmoothingAlt(self):
        """
        Metoda vrati pristup, ako vyhladit casovy rad. Napr. ak nastavim center, potom v pripade
        movingAverage metody pozadujem, aby v zavislosti na vyhladzovacom okne, vyhladzovacia hodnota
        bola v strede intervalu. To znamena, ze ak mam nastavene okno 5 hodnot, potom vezmem 2 hodnoty
        smerom do minulosti a 2 hodnoty smerom do buducnosti.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self._smoothingAlt

    def getVerbosity(self):
        """
        Funkcia vrati hodnotu verbosity pre popisnu statistiku. 0 znamena, ze sa nema nic zobrazovat.
        1 znamena, ze zobrazim vysledky vo vne triedy vo forme tabulky

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self._descVerbosity

    def getDescStations(self):
        """
        Funkcia vrati zoznam stanic, ktore by som chcel statisticky opisat. Bud zadam All a to znamena,
        ze postupne spracujem vsetky stanice, ktore sa nachadzaju

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self._descStations

    ###################################################################################################
    # protected funkcie

    def _loadConfig(self):
        """
        funkcia nacita konfiguracny subor.
        Returns
        -------
        None.
        """
        with open("config.json") as j:
            cf = json.load(j)

        # obecne nastavenie, napr. cesta/y k datam
        self._inpFileName = cf["setInput"]["inpFileName"]
        self._inpLocalPath = cf["setInput"]["inpLocalPath"]

        # prevezme nastavenie tykajuce sa intervalu spracovanej rady
        self._beg = cf["setInterval"]["beg"]
        self._end = cf["setInterval"]["end"]
        self._dbeg = cf["setInterval"]["dbeg"]
        self._dend = cf["setInterval"]["dend"]

        # prevezme nastavenie tykajuce sa Metody najmensich stvorcov
        self._maxIter = cf["setLSQ"]["maxIter"]
        self._epsVal = cf["setLSQ"]["epsVal"]

        # prevezme nastavenie tykajcue sa spracovania modelu
        self._limitRelativeUfG = cf["setModel"]["limitRelativeUfG"]
        self._calcWeights = cf["setModel"]["calcWeights"]
        self._addIntercept = cf["setModel"]["addIntercept"]
        self._probUp = cf["setModel"]["probUp"]

        # zoznam stanic a zoznam prenasobovacich konstant
        self._listStations = cf["setStations"]["stations"]

        # nacitanie nastavenia tykajuce sa vyhladenia casovych radov
        self._smoothingMethod = cf["setSmoothing"]["smoothingMethod"]
        self._smoothingBin = cf["setSmoothing"]["smoothingBin"]
        self._smoothingAlt = cf["setSmoothing"]["smoothingAlt"]

        # nacitanie nastavenia tykajuce sa vyberu stanice, ktoru chcem statisticky opisat.
        self._descStations = cf["setDescriptive"]["stations"]
        self._descVerbosity = cf["setDescriptive"]["verbosity"]
