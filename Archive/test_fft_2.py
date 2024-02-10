#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:56:00 2023

Subor funkcii je externych zdrojov a to konkrétne z [1] a [2]. Funkcie su ale postupne upravene tak, aby
vyhovovali mojim potrebam a poziadavkam.

[1] https://ataspinar.com/2020/12/22/time-series-forecasting-with-stochastic-signal-analysis-techniques/
[2] https://github.com/taspinar/siml/blob/master/siml/detect_peaks.py

@author: mel
"""

import numpy as np

import plotly.graph_objects as go
import datetime as dat
import math

import lib.support as sp
import lib.decoder as dc
import lib.config as cfg
import lib.MAD as md
import lib.SSA as ssa


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        # _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def construct_fft(yvalues, deg_polyfit=1, real_abs_only=True):
    N = len(yvalues)
    xvalues = np.arange(N)

    # we calculate the trendline and detrended signal with polyfit
    z2 = np.polyfit(xvalues, yvalues, deg_polyfit)
    p2 = np.poly1d(z2)
    yvalues_trend = p2(xvalues)
    yvalues_detrended = yvalues - yvalues_trend

    # The fourier transform and the corresponding frequencies
    fft_y = np.fft.fft(yvalues_detrended)
    fft_x = np.fft.fftfreq(N)
    if real_abs_only:
        fft_x = fft_x[:len(fft_x)//2]
        fft_y = np.abs(fft_y[:len(fft_y)//2])
    return fft_x, fft_y, p2


def get_integer_no_of_periods(yvalues, fft_x, fft_y, frac=1.0, mph=0.4):

    N = len(yvalues)
    fft_y_real = np.abs(fft_y[:len(fft_y)//2])
    fft_x_real = fft_x[:len(fft_x)//2]

    mph = np.nanmax(fft_y_real)*mph
    indices_peaks = detect_peaks(fft_y_real, mph=mph)
    peak_fft_x = fft_x_real[indices_peaks]
    main_peak_x = peak_fft_x[0]
    T = int(1/main_peak_x)

    no_integer_periods_all = N//T
    no_integer_periods_frac = int(frac*no_integer_periods_all)
    no_samples = T*no_integer_periods_frac

    yvalues_ = yvalues[-no_samples:]
    xvalues_ = np.arange(len(yvalues))
    return xvalues_, yvalues_


def restore_signal_from_fft(fft_x, fft_y, N, extrapolate_with, frac_harmonics):

    xvalues_full = np.arange(0, N + extrapolate_with)
    restored_sig = np.zeros(N + extrapolate_with)
    indices = list(range(N))

    # The number of harmonics we want to include in the reconstruction
    indices.sort(key=lambda i: np.absolute(fft_x[i]))
    # ak chcem vyhladit po najvyznamnejsich frekvenciach, tak potom zakomentovat r207. Ak dam fft_y, tak
    # vyhladim po najvyznamnejsich amplitudach
    # indices.reverse()
    max_no_harmonics = len(fft_y)
    no_harmonics = int(frac_harmonics*max_no_harmonics)

    for i in indices[:1 + no_harmonics * 2]:
        ampli = np.absolute(fft_y[i]) / N
        phase = np.angle(fft_y[i])
        restored_sig += ampli * np.cos(2 * np.pi * fft_x[i] * xvalues_full + phase)
    # return the restored signal plus the previously calculated trend
    return restored_sig


def reconstruct_from_fft(yvalues,
                         frac_harmonics=1.0,
                         deg_polyfit=2,
                         extrapolate_with=0,
                         fraction_signal=1.0,
                         mph=0.4):

    # N_original = len(yvalues)
    fft_x, fft_y, p2 = construct_fft(yvalues, deg_polyfit, real_abs_only=False)
    xvalues, yvalues = get_integer_no_of_periods(yvalues, fft_x, fft_y, frac=fraction_signal, mph=mph)
    fft_x, fft_y, p2 = construct_fft(yvalues, deg_polyfit, real_abs_only=False)
    N = len(yvalues)

    xvalues_full = np.arange(0, N + extrapolate_with)
    restored_sig = restore_signal_from_fft(fft_x, fft_y, N, extrapolate_with, frac_harmonics)
    restored_sig = restored_sig + p2(xvalues_full)

    return restored_sig[-extrapolate_with:]


def main():
    signal = np.array([669, 592, 664, 1005, 699, 401, 646, 472, 598, 681, 1126, 1260, 562, 491, 714, 530, 521,
                       687, 776, 802, 499, 536, 871, 801, 965, 768, 381, 497, 458, 699, 549, 427, 358, 219,
                       635, 756, 775, 969, 598, 630, 649, 722, 835, 812, 724, 966, 778, 584, 697, 737, 777,
                       1059, 1218, 848, 713, 884, 879, 1056, 1273, 1848, 780, 1206, 1404, 1444, 1412, 1493,
                       1576, 1178, 836, 1087, 1101, 1082, 775, 698, 620, 651, 731, 906, 958, 1039, 1105, 620,
                       576, 707, 888, 1052, 1072, 1357, 768, 986, 816, 889, 973, 983, 1351, 1266, 1053, 1879,
                       2085, 2419, 1880, 2045, 2212, 1491, 1378, 1524, 1231, 1577, 2459, 1848, 1506, 1589,
                       1386, 1111, 1180, 1075, 1595, 1309, 2092, 1846, 2321, 2036, 3587, 1637, 1416, 1432,
                       1110, 1135, 1233, 1439, 894, 628, 967, 1176, 1069, 1193, 1771, 1199, 888, 1155, 1254,
                       1403, 1502, 1692, 1187, 1110, 1382, 1808, 2039, 1810, 1819, 1408, 803, 1568, 1227,
                       1270, 1268, 1535, 873, 1006, 1328, 1733, 1352, 1906, 2029, 1734, 1314, 1810, 1540,
                       1958, 1420, 1530, 1126, 721, 771, 874, 997, 1186, 1415, 973, 1146, 1147, 1079, 3854,
                       3407, 2257, 1200, 734, 1051, 1030, 1370, 2422, 1531, 1062, 530, 1030, 1061, 1249, 2080,
                       2251, 1190, 756, 1161, 1053, 1063, 932, 1604, 1130, 744, 930, 948, 1107, 1161, 1194,
                       1366, 1155, 785, 602, 903, 1142, 1410, 1256, 742, 985, 1037, 1067, 1196, 1412, 1127,
                       779, 911, 989, 946, 888, 1349, 1124, 761, 994, 1068, 971, 1157, 1558, 1223, 782, 2790,
                       1835, 1444, 1098, 1399, 1255, 950, 1110, 1345, 1224, 1092, 1446, 1210, 1122, 1259,
                       1181, 1035, 1325, 1481, 1278, 769, 911, 876, 877, 950, 1383, 980, 705, 888, 877, 638,
                       1065, 1142, 1090, 1316, 1270, 1048, 1256, 1009, 1175, 1176, 870, 856, 860])

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
    signal = F_ssa.reconstruct(slice(0, S))

    t = np.arange(0, signal.size)
    sig = reconstruct_from_fft(vals_data, frac_harmonics=0.02)

    fo = go.Figure()
    fo.add_trace(go.Scatter(x=t, y=vals_data, mode="lines"))
    fo.add_trace(go.Scatter(x=t, y=sig, mode="lines"))
    fo.show()


if __name__ == "__main__":
    main()
