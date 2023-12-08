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

    def getAlternativeModel(self):
        """
        Funkcia vrati prepinac medzi alternativnym True model a regularnym False modelom

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self._alternativeModel

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

    def getPlotSmoothingResults(self):
        """
        Ak je v config nastaveny True, resp. 1, potom vygenerujem obrazky validne pre SSA metodu
        """

        return self._plotSmoothingResults

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

    def getDescVerbosity(self):
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

    def getValidation(self):
        """
        Funkcia vrati prepinac validation. To znamena, ze ak je True, pozadujem validaciu modela. Ak False,
        validaciu nepozadujem. Pozor, treba sa rozhodnut s ohladom na time span.

        Returns
        -------
        _validation : TYPE
            DESCRIPTION.

        """

        return self._validation

    def getPrediction(self):
        """
        Funkcia vrati priznak prediction. Ten je validny, iba ak je validation Trie. Ak je prediction True,
        to znamena, ze pozadujem zhodnotit model tak, ze vezmem poslednu cast roka alebo nejaku cast dat a tu
        extrapolujem pomocou ziskanych konstant nasho modelu. Potom vysledkom su spocitane metriky, ktore
        ukazuju na kvalitu validacie. Ak je prediction False, potom pozadujem vybrat X% nahodnych bodov v
        modele, ich hodnoty odhadnem pomocou modelu a porovnam ich s hodnotami, ktore som si bokom odlozil.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._prediction

    def getValidchart(self):
        """
        Metoda vrati priznak, podla ktoreho sa rozhodnem, ze ci v triede validation budem aj kreslit. Trocha
        sa mi to nepozdava z toho pohladu, ze trieda by mala validovat a na kreslenie by mal byt priestor v
        inej casti kodu. Ale z praktickeho hladiska sa to hodi, a mam tam vsetky data. Plotitko bude impleme-
        tovane v triede charts.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._validchart

    def getOutLocalPath(self):
        """
        Metoda vrati lokalnu adresu na mieto, kam sa budu ukladat vysledky

        Returns
        -------
        None.

        """

        return self._outLocalPath

    def getFigFolderName(self):
        """
        Nazov adresara, kam sa bud ukladat obrazky

        Returns
        -------
        None.

        """

        return self._figFolderName

    def getCsvFolderName(self):
        """
        Nazov adresara, kam sa budu ukladat CSV subory

        Returns
        -------
        None.

        """

        return self._csvFolderName

    def getLinearity(self):
        """
        Prepinatko: True, chcem zobrazit pomer UfG a Toku a pozriet sa na scatter, kde je zobrazeny vztach
        medzi datami

        Returns
        -------
        None.

        """

        return self._linearity

    def getGroups(self):
        """
        Prepinatko: True, chcem analyzovat data podla skupin, naprp. YOY (year over year) alebo WOW (week
        over week)

        Returns
        -------
        None.

        """

        return self._groups

    def getVerbosityLSQ(self):
        """
        Vrati koeficient verbosity pre printy na plochu. 0 - nic nevypisuj. 1 - Len strucne info (asi koef a
        stredne hodnoty, pripadne aj vysledky vyrovnania a 2 - podrobne info)

        Returns
        -------
        None.

        """

        return self._verLSQ

    def getCentering(self):
        """
        Ak je v nastaveni True/1, to znamena, ze v decpder.py funkcii sa postaram o to, aby som od stlpcov
        odcital ich priemer a data tak centroval na nulu

        Returns
        -------
        None.

        """

        return self._centering

    def getInvestLin(self):
        """
        Ak je config nastaveny na True, potom v decoder.py funkcii sa zaoberam studovanim linearnej zavoslosti
        medzi jednotliv7mi stlpcami
        """

        return self._investLin

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

        # nastavenie vstupu, napr. cesta/y k datam
        self._inpFileName = cf["setInput"]["inpFileName"]
        self._inpLocalPath = cf["setInput"]["inpLocalPath"]

        # nastavenie vytupu
        self._outLocalPath = cf["setOutput"]["outLocalPath"]
        self._figFolderName = cf["setOutput"]["figFolderName"]
        self._csvFolderName = cf["setOutput"]["csvFolderName"]

        # prevezme nastavenie tykajuce sa intervalu spracovanej rady
        self._beg = cf["setInterval"]["beg"]
        self._end = cf["setInterval"]["end"]
        self._dbeg = cf["setInterval"]["dbeg"]
        self._dend = cf["setInterval"]["dend"]

        # prevezme nastavenie tykajuce sa spracovania dat este v dekodery
        self._centering = cf["setDecoder"]["centering"]
        self._investLin = cf["setDecoder"]["investLin"]

        # prevezme nastavenie tykajuce sa Metody najmensich stvorcov
        self._maxIter = cf["setLSQ"]["maxIter"]
        self._epsVal = cf["setLSQ"]["epsVal"]
        self._verLSQ = cf["setLSQ"]["verbosity"]

        # prevezme nastavenie tykajcue sa spracovania modelu
        self._alternativeModel = cf["setModel"]["alternativeModel"]
        self._limitRelativeUfG = cf["setModel"]["limitRelativeUfG"]
        self._calcWeights = cf["setModel"]["calcWeights"]
        self._addIntercept = cf["setModel"]["addIntercept"]
        self._probUp = cf["setModel"]["probUp"]

        # prevezme nastavenie validacie modela
        self._validation = cf["setValidation"]["validation"]
        self._prediction = cf["setValidation"]["prediction"]
        self._validchart = cf["setValidation"]["validchart"]

        # zoznam stanic a zoznam prenasobovacich konstant
        self._listStations = cf["setStations"]["stations"]

        # nacitanie nastavenia tykajuce sa vyhladenia casovych radov
        self._smoothingMethod = cf["setSmoothing"]["smoothingMethod"]
        self._plotSmoothingResults = cf["setSmoothing"]["plotResults"]
        self._smoothingBin = cf["setSmoothing"]["smoothingBin"]
        self._smoothingAlt = cf["setSmoothing"]["smoothingAlt"]

        # nacitanie nastavenia tykajuce sa vyberu stanice, ktoru chcem statisticky opisat.
        self._descStations = cf["setDescriptive"]["stations"]
        self._descVerbosity = cf["setDescriptive"]["verbosity"]

        # nacitanie nastavenia linearity, t.j. zobrazenie ufg vs. flow
        self._linearity = cf["setVisualisations"]["relations"]
        self._groups = cf["setVisualisations"]["groups"]
