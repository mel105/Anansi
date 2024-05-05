#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 2023

Skript je urceny k tomu, aby som odvodil, implementoval, ladil tzv Model A.
Model A je 'blind' model potrebny na odhad a predpoved parametru strat plynu.
Model je zalozeny na predpoklade, ze k simulovaniu strat nepotrebujeme odhadnut
koeficienty, ktore opisuju straty na jednotlivych staniciach, ale potrebuje len
realne straty, ktore su definovane:

UfGr = Suma(Exit)-Sima(Intakes)+dLP,

kde
    EXIT je suhrn tokov na vystupe,
    INTAKES je suhrn tokov na vstupe a
    dLP rozdiel akumulacie medzi dvoma plynarenskymi dnami.

Riesenie modelu je zalozeny na linearnej regresii pravdepodobne polynomickej funkcie.
K odhadu koeficientov tochto modelu sa zrejme pouzije Metoda najmensich stvorcov.
Vzhladom k tomu, ze k simulovaniu a rieseniu problemov pouzivame moduly programu
Python, pouziju sa add-in funkcie tohto programu.

Postupne by tento skript ma zastresit tieto ukony:
    1. Nacita data
    2. Odstrani outliers
    3. Vyhladi data (redukuje sa vplyv nepresnosti odhadu akumulacie)
    4. Modeluje straty
    5. Validacia modelu
    5. Odhad strat do buducnosti (tzv. forecast

    author: mel
"""

import plotly.io as pio
import warnings
import math
import lib.support as sp

import lib.config as cfg

import lib.SSA as ssa


import sys

import numpy as np
import datetime as dat
import pathlib
import os

import lib.process_model_A as ma

np.random.seed(1234)

pio.renderers.default = 'browser'

warnings.filterwarnings("ignore")


def run():

    # Nacitanie koniguracneho suboru
    confObj = cfg.config()

    # Nastavenie adresarov, kam sa budu ukladat obrazky/subory CSV (neskor treba textove protokoly)
    sp.checkFolder(confObj.getOutLocalPath())
    sp.checkFolder(confObj.getOutLocalPath()+"/"+confObj.getFigFolderName())
    sp.checkFolder(confObj.getOutLocalPath()+"/"+confObj.getCsvFolderName())

    # Spracovanie dat zavisi na volbe modelu. Takze v prvom rade si z konfiguraku nacitam volbu modelu a podla
    # zvoleneho modelu spracujem data
    if confObj.getModel() == "A":
        print("Model A is processed")
        ma.process_Model_A(confObj)

    elif confObj.getMaxIter() == "B":
        print("Model B is processed")
    else:
        print("Required model is not implemented yet")
        sys.exit()

    return 0


# Spustenie spracovania dat
if __name__ == "__main__":

    run()
