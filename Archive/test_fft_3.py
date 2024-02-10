#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:55:51 2023

@author: mel
"""

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


# from PRML.prml.preprocess import PolynomialFeature
# from PRML.prml.linear import (
#     LinearRegression,
#     RidgeRegression,
#     BayesianRegression
# )

import plotly.io as pio
import warnings
import math
import lib.support as sp
import lib.decoder as dc
import lib.config as cfg
import lib.MAD as md
import lib.SSA as ssa
import lib.charts as cha
import pylab as pl
import numpy as np
import datetime as dat
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.fft import fft, fftfreq, fftshift
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
    signal = ssa_data
    n = len(signal)
    t = np.arange(0, n)
    dt = 1

    fhat = np.fft.fft(signal, n)  # computes the fft
    psd = fhat * np.conj(fhat)/n
    freq = (1/(dt*n)) * np.arange(n)  # frequency array
    idxs_half = np.arange(1, np.floor(n/2), dtype=np.int32)  # first half index

    # Filter out noise
    threshold = 20e12
    psd_idxs = psd > threshold  # array of 0 and 1
    psd_clean = psd * psd_idxs  # zero out all the unnecessary powers
    fhat_clean = psd_idxs * fhat  # used to retrieve the signal

    signal_filtered = np.fft.ifft(fhat_clean)  # inverse fourier transform

    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(x=freq[idxs_half], y=np.abs(psd[idxs_half]), mode="lines"))
    fig0.show()

    # Visualization
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(t, signal, color='b', lw=0.5, label='Analysed Signal')
    #ax[0].plot(t, signal_clean, color='r', lw=1, label='Clean Signal')
    #ax[0].set_ylim([minsignal, maxsignal])
    ax[0].set_xlabel('t axis')
    ax[0].set_ylabel('Vals')
    ax[0].legend()

    ax[1].plot(freq[idxs_half], np.abs(psd[idxs_half]), color='b', lw=0.5, label='PSD noisy')
    ax[1].set_xlabel('Frequencies in Hz')
    ax[1].set_ylabel('Amplitude')
    ax[1].legend()

    ax[2].plot(freq[idxs_half], np.abs(psd_clean[idxs_half]), color='r', lw=1, label='PSD clean')
    ax[2].set_xlabel('Frequencies in Hz')
    ax[2].set_ylabel('Amplitude')
    ax[2].legend()

    ax[3].plot(t, signal_filtered, color='r', lw=1, label='Clean Signal Retrieved')
    #ax[3].set_ylim([minsignal, maxsignal])
    ax[3].set_xlabel('t axis')
    ax[3].set_ylabel('Vals')
    ax[3].legend()

    plt.subplots_adjust(hspace=0.4)
    plt.savefig('signal-analysis.png', bbox_inches='tight', dpi=300)
    fig.show()

    """
    xo = ssa_data
    Ts = 24*60*60  # time sample. Vzorkovaci cas. Mame k dispo denne data. Den je tu konvertovany na second
    t = time_data_s  # cas. Denne data v sekundach

    Ts = 1
    t = np.arange(0, time_data_s.size)  # cas. Denne data v indexe.

    # detrend
    p = np.polyfit(t, xo, 1)
    xnot = xo - p[0] * t  # data ocistene o trend
    n = len(xnot)
    # Hamming
    xnot = xnot * np.hamming(len(xnot))  # pre potlacenie skreslenia frekvencnej charakteristiky

    # Fourierova transformace
    yf = fft(xnot.to_numpy())  # amplitudy, t.j. data sso ocistene o trend vo frekvencom spektre
    fs = 1/Ts  # vzorkovacia frekvencia. vzorkovaci cas vo frekvencnom spektre
    xf = fftfreq(len(t), d=fs)   # frekvencie

    N = n // 2

    f = []
    for i in range(0, len(yf)):
        f.append(i*fs/len(yf))

    ffto = go.Figure()
    # ffto.add_trace(go.Scatter(x=xf[:N], y=abs(yf[:N]), mode="lines"))
    ffto.add_trace(go.Scatter(x=np.arange(0, N), y=np.absolute(yf[:N]), mode="lines"))
    ffto.show()

    # n = len(xnot)
    # fshift = []
    # for i in np.arange(-n/2, n/2-1, 1):
    #     fshift.append(i*fs/n)

    # yshift = fftshift(yf)
    # # ymag = 10*np.log10(np.abs(yshift)**2)

    # # fo3 = go.Figure()
    # # fo3.add_trace(go.Scatter(x=fshift, y=ymag, mode="lines"))
    # # fo3.show()

    # # vezmem len spektrum od nuly do +

    fos = xf[:N]
    tos = 1/np.array(fos)  # * (24*60*60)
    ymagos = np.absolute(yf[:N])

    fo4 = go.Figure()
    fo4.add_trace(go.Scatter(x=tos, y=ymagos, mode="lines"))
    fo4.show()

    # # Spracovanie forecastu
    restored_sig = fft_reconstruction(t, ymagos, xf[:N])

    # # pridanie trendu
    restored_sig = restored_sig + p[0] * t

    foh = go.Figure()
    foh.add_trace(go.Scatter(x=t, y=xo, mode="lines"))
    foh.add_trace(go.Scatter(x=t, y=restored_sig, mode="lines"))
    foh.show()
    """
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
    # test()
    # testSignalProcessing()
