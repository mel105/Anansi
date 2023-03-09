#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:50:03 2023

@author: mel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plotScatter(lVec, eVec):
    """
    Funkcia sluzi k vyploteniu scatter plotu. V titulke bude mat hodnotu korelacneho koeficientu

    Parameters
    ----------
    lVec : TYPE
        DESCRIPTION.
    eVec : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    figScatter = plt.figure()

    ax = figScatter.add_subplot(111)
    ax.scatter(lVec, eVec)

    plt.xlabel("Real UfG",  fontsize=20)
    plt.ylabel("Model UfG", fontsize=10)
    plt.title("$Correlation =$", size=18)
    plt.grid(alpha=.4)

    # save the plots
    plt.savefig('scatter.png', dpi=300)

    plt.show()


def plotHistogram(res):
    """
    Funkcia vykresli histogram

    Parameters
    ----------
    res : TYPE numpy.array
        DESCRIPTION. Vektor, ktory obsahuje rozdiel medzi modelom a meraniami

    Returns
    -------
    None.

    """

    # Statistiky pre vykreslenie normalneho rozdelenia
    # Mean and standard deviation
    mu, std = norm.fit(res)

    # Vykreslenie histogramu

    n, bins, patches = plt.hist(res, bins='auto', facecolor='#0504aa', alpha=0.7, rwidth=0.85)

    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("UfG: [real - model]")
    plt.ylabel("Frequency")

    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    # Vykreslenie normalneho rozdelenia
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    s = norm.pdf(x, mu, std)

    area_hist = .0
    for ii in range(n.size):
        area_hist += (bins[ii+1]-bins[ii]) * n[ii]

    # oplot fit into histogram
    plt.plot(x, s*area_hist, label='fitted and area-scaled PDF', linewidth=4)
    plt.legend()
    title = "Fit Values: MEAN: {:.2f} and SDEV: {:.2f}".format(mu, std)
    plt.title(title, fontsize=10)

    plt.show()


def plotDetails(date, lvec, valEst, beg, end):
    """
    Funkcia zobrazi detail casovej rady
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    lvec : TYPE
        DESCRIPTION.
    valEst : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """

    df = pd.DataFrame()
    df["DATE"] = date.to_list()
    df["REF"] = lvec.tolist()
    df["EST"] = valEst.tolist()

    # filtrovanie dat podla zadanych intervalov
    df = df[(df['DATE'] >= beg) & (df['DATE'] <= end)]

    x = df["DATE"]
    y1 = df["REF"]
    y2 = df["EST"]

    fig = plt.figure()
    fig.show()
    ax = fig.add_subplot(111)

    ax.plot(x, np.stack(y1).astype(None), color="tab:blue", ls="-", label="Real UfG")
    ax.plot(x, np.stack(y2).astype(None), color="tab:red", ls="-", label="Model UfG")
    plt.xlabel("Date",  fontsize=20)
    plt.xticks(rotation=30, ha='right', size=12)
    plt.ylabel("$Delta$", fontsize=10)
    plt.title("$Delta = Offtakes-Intakes$", size=18)
    plt.grid(alpha=.4)
    plt.legend(loc=2)

    # save the plots
    plt.savefig('vyrovnanie_detail.png', dpi=300)


def plotResults(timeVec, lVec, valEst):
    """
    Funkcia vykresli zakladne zobrazenie realneho UfG a modelovaneho na zakladne LSQ vyrovnania

    Parameters
    ----------
    df : TYPE dataframe
        DESCRIPTION. obsahuje vsetky data. Asi by mi stacil len vektor DATE
    lVec : TYPE numpy.mnarray
        DESCRIPTION. lVec obsahuje realne UfG
    valEst : TYPE numpy.ndarray
        DESCRIPTION. Obsahuje modelovane UfG

    Returns
    -------
    None.

    """

    # y = df["DATE"]

    fig = plt.figure()
    fig.show()
    ax = fig.add_subplot(111)

    ax.plot(timeVec, lVec, color="tab:blue", ls="-", label="Real UfG")
    ax.plot(timeVec, valEst, color="tab:red", ls="-", label="Model UfG")
    plt.xlabel("Date",  fontsize=20)
    plt.xticks(rotation=30, ha='right', size=12)
    plt.ylabel("$Delta$", fontsize=10)
    plt.title("$Delta = Offtakes-Intakes$", size=18)
    plt.grid(alpha=.4)
    plt.legend(loc=2)

    # save the plots
    plt.savefig('vyrovnanie.png', dpi=300)

    plt.draw()
