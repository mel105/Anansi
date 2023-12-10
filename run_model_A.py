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
import lib.approx_fft as aprx
import lib.visualApprox as va
import lib.forecast as frc

import numpy as np
import datetime as dat
import pathlib
import os

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

    # Nacitanie realnych strat
    fileName = confObj.getInpFileName()
    filePath = confObj.getInpFilePath()
    decObj = dc.decoder(fileName, filePath, confObj)

    time_data = decObj.getDF()["DATE"]
    time_data_s = time_data - dat.datetime(1900, 1, 1)
    time_data_s = time_data_s.dt.total_seconds()
    time_data_s = time_data_s - time_data_s[0]
    vals_data = decObj.getDF()["UFGr"]

    # ######################################################### OUTLIERS
    # Odhad a eliminovanie realnych odlahlych pozorovani
    vals_median, MAD, lowIdx, uppIdx = md.MAD(vals_data)
    # fo = md.plot(time_data, vals_data, lowIdx, uppIdx)
    # fo.show()

    out_data = md.replacement(vals_data, lowIdx, uppIdx, vals_median)
    # fo = md.plot(time_data, out, lowIdx, uppIdx)
    # fo.show()

    # ######################################################### SSA SMOOTHING
    # Vyhladenie redukovaneho casoveho radu
    L = math.floor(decObj.getDF().shape[0]/2)  # max hodnota okna, v ktorom sa vysetruju vlastne vektory
    F_ssa = ssa.SSA(out_data, L)

    S = 200  # vezmem si prvych S vlastnych vektorov, pomocou ktrorych zrekonstruujem orig. casovu radu
    ssa_data = F_ssa.reconstruct(slice(0, S))

    # pokracovanie rekonstrukcie
    # noise = F_ssa.reconstruct(slice(S+1, L))
    # origs = F_ssa.orig_TS
    # cha.ssaPlot(time_data, origs, ssa_data, noise)

    # ######################################################### FFT APPROXIMATION
    sig, fft_x_orig, fft_y_orig, fft_x_r, fft_y_r, fft_x, fft_y, _, _, peaks, indexes, noHarm, trd =\
        aprx.reconstruct_from_fft(ssa_data, frac_harmonics=0.02)

    # ######################################################### FORECASTING
    n_future = 365  # pocet dni, ktore chcem predikovat

    sumsOverYears, sumsOverMonths, time_data_ex = frc.forecast(time_data, ssa_data, n_future, indexes, noHarm,
                                                               fft_x, fft_y, trd)

    # 4, suhrna tabulka, kde su nejake statistiky a treba prediction for last 50 days in total is XXX kWh

    actualPath = pathlib.Path(__file__).parent.resolve()
    os.makedirs(actualPath, exist_ok=True)
    print(sumsOverYears)
    sumsOverYears.to_csv(str(actualPath)+"/results/CSV/sumsOverYears.csv")

    print(sumsOverMonths)
    sumsOverMonths.to_csv(str(actualPath)+"/results/CSV/sumsOverMonths.csv")

    # ######################################################### RESULTS VISUALISATION
    fo1, yvalues_detrended = va.plot_data(time_data, ssa_data)  # original data visualisation
    fo1.write_image(str(actualPath)+"/results/FIG/original.png", width=1000, height=650, scale=2)
    # fo1.show()

    fo2 = va.plot_fft(yvalues_detrended, fft_x_r, fft_y_r, peaks)  # results of FFT presentation
    fo2.write_image(str(actualPath)+"/results/FIG/fft.png", width=1000, height=650, scale=2)
    # fo2.show()

    fo3 = va.plot_approx(time_data, ssa_data, sig)  # results of approximation presentation
    fo3.write_image(str(actualPath)+"/results/FIG/approx.png", width=1000, height=650, scale=2)
    # fo3.show()

    fo4 = va.plot_forecast(time_data_ex, ssa_data)  # forecast presentaiton
    fo4.write_image(str(actualPath)+"/results/FIG/forecast.png", width=1000, height=650, scale=2)
    # fo4.show()

    return decObj, confObj


# Spustenie spracovania dat
if __name__ == "__main__":

    data,  conf = run()
