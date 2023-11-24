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
import lib.decoder as dc
import lib.config as cfg
import lib.MAD as md
import lib.SSA as ssa
import lib.charts as cha

import numpy as np
import datetime as dat
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.fft import fft, fftfreq

# from PRML.prml.preprocess import PolynomialFeature
# from PRML.prml.linear import (
#     LinearRegression,
#     RidgeRegression,
#     BayesianRegression
# )

np.random.seed(1234)

pio.renderers.default = 'browser'

warnings.filterwarnings("ignore")


def test():

    x_train, y_train = create_toy_data(func, 1000, 0.1)
    x_test = np.linspace(0, 1, 100)
    y_test = func(x_test)

    err = []
    dg = []
    for i in range(100):

        poly = np.polyfit(x_test, y_test, deg=i)
        y_approx = np.polyval(poly, x_test)
        dg.append(i)
        err.append(rmse(y_test, y_approx))

    fo = go.Figure()
    fo.add_trace(go.Scatter(x=x_train, y=y_train, mode="markers"))
    fo.add_trace(go.Scatter(x=x_test, y=y_test, mode="lines"))
    fo.add_trace(go.Scatter(x=x_test, y=y_approx, mode="lines"))

    fo.show()

    fo1 = go.Figure()
    fo1.add_trace(go.Scatter(x=dg, y=err, mode="lines+markers"))
    fo1.show()

    return 0


def create_toy_data(func, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t


def func(x):
    return np.sin(2 * np.pi * x/4)


def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))


# OSTRA FUNKCE

def run():

    # Nacitanie koniguracneho suboru
    confObj = cfg.config()

    # Nastavenie adresarov, kam sa budu ukladat obrazky/subory CSV (neskor treba textove protokoly)
    sp.checkFolder(confObj.getOutLocalPath())
    sp.checkFolder(confObj.getOutLocalPath()+"/"+confObj.getFigFolderName())
    sp.checkFolder(confObj.getOutLocalPath()+"/"+confObj.getCsvFolderName())

    # Nacitanie realnych strat
    fileName = confObj.getInpFileName()
    filePath = confObj.getInpFilePath()
    decObj = dc.decoder(fileName, filePath, confObj)

    time_data = decObj.getDF()["DATE"]
    time_data_s = time_data - dat.datetime(1900, 1, 1)
    time_data_s = time_data_s.dt.total_seconds()
    time_data_s = time_data_s - time_data_s[0]
    vals_data = decObj.getDF()["UFGr"]

    # Odhad a eliminovanie realnych odlahlych pozorovani
    vals_median, MAD, lowIdx, uppIdx = md.MAD(vals_data)
    # fo = md.plot(time_data, vals_data, lowIdx, uppIdx)
    # fo.show()

    out_data = md.replacement(vals_data, lowIdx, uppIdx, vals_median)
    # fo = md.plot(time_data, out, lowIdx, uppIdx)
    # fo.show()

    # Vyhladenie redukovaneho casoveho radu
    L = math.floor(decObj.getDF().shape[0]/2)  # max hodnota okna, v ktorom sa vysetruju vlastne vektory
    F_ssa = ssa.SSA(out_data, L)

    S = 200  # vezmem si prvych S vlastnych vektorov, pomocou ktrorych zrekonstruujem orig. casovu radu
    ssa_data = F_ssa.reconstruct(slice(0, S))

    # pokracovanie rekonstrukcie
    # noise = F_ssa.reconstruct(slice(S+1, L))
    # origs = F_ssa.orig_TS
    # cha.ssaPlot(time_data, origs, ssa_data, noise)

    # Spracovanie modelu
    deg_vec = []
    err_vec = []

    for deg_index in range(16):
        print(deg_index)
        poly = np.polyfit(time_data_s, ssa_data, deg=deg_index)

        y_approx = np.polyval(poly, time_data_s)
        deg_vec.append(deg_index)
        err_vec.append(rmse(ssa_data, y_approx))

    # Validacia modelu
    fo = go.Figure()
    fo.add_trace(go.Scatter(x=time_data, y=ssa_data, mode="lines"))
    fo.add_trace(go.Scatter(x=time_data, y=y_approx, mode="lines"))
    fo.show()

    fo1 = go.Figure()
    fo1.add_trace(go.Scatter(x=deg_vec, y=err_vec, mode="lines+markers"))
    fo1.show()

    # fourierova transformace
    yf = fft(ssa_data)
    N = L*2
    T = 1
    xf = fftfreq(N, T)[:N//2]

    fo2 = go.Figure()
    fo2.add_trace(go.Scatter(x=xf, y=2.0/N * np.abs(yf[0:N//2]), mode="lines"))
    fo2.show()

    # Spracovanie forecastu

    return decObj, confObj


# Spustenie spracovania dat
if __name__ == "__main__":

    data,  conf = run()
    # test()
