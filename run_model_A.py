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
from scipy.signal import welch

import numpy as np
import datetime as dat
import plotly.graph_objects as go

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
    sig, fft_x_orig, fft_y_orig, fft_x_r, fft_y_r, fft_x, fft_y, _, _, peaks =\
        aprx.reconstruct_from_fft(ssa_data, frac_harmonics=0.02)

    # ######################################################### FORECASTING
    n_future = 50  # pocet dni, ktore chcem predikovat

    # suhrna tabulka, kde su nejake statistiky a treba prediction for last 50 days in total is XXX kWh

    # ######################################################### RESULTS VISUALISATION
    # Zobrazenie orig dat bez trendu a trend samostatne
    z2 = np.polyfit(np.arange(0, ssa_data.size), ssa_data, 2)
    p2 = np.poly1d(z2)
    yvalues_trend = p2(np.arange(0, ssa_data.size))
    yvalues_detrended = ssa_data - yvalues_trend

    fo1 = go.Figure()
    fo1.add_trace(go.Scatter(x=time_data, y=yvalues_detrended, name="Analysed data", mode="lines",
                             line=dict(color="royalblue", width=3)))
    fo1.add_trace(go.Scatter(x=time_data, y=yvalues_trend, name="Trend", mode="lines",
                             line=dict(color="black", width=4, dash='dot')))
    fo1.update_layout(title='Presentation of de-trended analysed data',
                      xaxis_title="Time #day",
                      yaxis_title="Real values [kWh]",
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=1.02,
                          xanchor="right",
                          x=1
                      )
                      )
    # fo1.show()

    # Zobrazenie frekvencneho spektra a PSD (Power Spectral Density)
    psd_x, psd_y = welch(yvalues_detrended)
    peak_fft_x, peak_fft_y = fft_x_r[peaks], fft_y_r[peaks]

    fo2 = go.Figure()
    fo2.add_trace(go.Scatter(x=fft_x_r, y=fft_y_r, name="FFT", mode="lines",
                             line=dict(color="royalblue", width=2)))
    fo2.add_trace(go.Scatter(x=psd_x, y=np.sqrt(psd_y)*100, name="PSD", mode="lines",
                             line=dict(color="red", width=2, dash='dot')))
    fo2.update_layout(title='Presentation of Fourier frequency spectrum as well as Power density spectrum',
                      xaxis_title="Frequency [Hz]",
                      yaxis_title="Spectrum",
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=1.02,
                          xanchor="right",
                          x=1
                      )
                      )

    text_list = []
    xl_list = []
    yl_list = []
    for ii in range(len(peaks)):
        xl, yl = peak_fft_x[ii], peak_fft_y[ii]
        T = 1/xl
        text_label = "  f = {:.2f}\n  T = {:.2f}".format(xl, T)
        xl_list.append(xl)
        yl_list.append(yl)
        text_list.append(text_label)

    fo2.add_trace(go.Scatter(x=xl_list, y=yl_list, mode="text", text=text_list,
                             textposition="top center"))
    # fo2.show()

    # Zobrazenie aproximacie casovej rady pomocou FFT
    fo3 = go.Figure()
    fo3.add_trace(go.Scatter(x=time_data, y=ssa_data, name="Analysed time series", mode="lines",
                             line=dict(color="royalblue", width=3)))
    fo3.add_trace(go.Scatter(x=time_data, y=sig, name="Reconstructed time series", mode="lines",
                             line=dict(color="tomato", width=3)))
    fo3.update_layout(title="Presentation of analysed time series vs reconstructed time series with \n" +
                      "application of inverse Fourier transform",
                      xaxis_title="Time #day",
                      yaxis_title="Real values [kWh]",
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=1.02,
                          xanchor="right",
                          x=1
                      )
                      )

    # fo3.show()

    return decObj, confObj


def fft_reconstruction(t, am, f, n_harm=50):

    n = f.size
    # t = np.arange(0, n)

    indexes = list(range(n))

    # sort indexes by amplitudes, higher to lower
    indexes.sort(key=lambda i: np.absolute(am[i]))
    indexes.reverse()

    restored_sig = np.zeros(t.size)

    for i in indexes[:1+n_harm]:

        ampli = np.absolute(am[i]) / n   # amplitude
        phase = np.angle(am[i])          # phase

        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig


# Spustenie spracovania dat
if __name__ == "__main__":

    data,  conf = run()
