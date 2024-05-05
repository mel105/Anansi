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
    return xvalues_, yvalues_, fft_x_real, fft_y_real, indices_peaks


def restore_signal_from_fft(fft_x, fft_y, N, extrapolate_with, frac_harmonics=20):

    xvalues_full = np.arange(0, N + extrapolate_with)
    restored_sig = np.zeros(N + extrapolate_with)
    indices = list(range(N))

    # The number of harmonics we want to include in the reconstruction

    indices.sort(key=lambda i: np.absolute(fft_y[i]))
    # ak chcem vyhladit po najvyznamnejsich frekvenciach, tak potom zakomentovat r205. Ak dam fft_y, tak
    # vyhladim po najvyznamnejsich amplitudach
    indices.reverse()

    # max_no_harmonics = len(fft_y)
    no_harmonics = frac_harmonics  # int(frac_harmonics*max_no_harmonics)

    for i in indices[:1 + no_harmonics * 2]:

        ampli = np.absolute(fft_y[i]) / N
        phase = np.angle(fft_y[i])
        restored_sig += ampli * np.cos(2 * np.pi * fft_x[i] * xvalues_full + phase)

    # return the restored signal plus the previously calculated trend
    return restored_sig, indices, no_harmonics


def reconstruct_from_fft(yvalues,
                         frac_harmonics=10,
                         deg_polyfit=2,
                         extrapolate_with=0,
                         fraction_signal=1.0,
                         mph=0.22):
    """
    Hlavna funkcia, ktora sluzi k spracovaniu casoveho radu pomocou FFT a nasledne inverzna transformacia je
    spocitana pomocou harmonickej cosinu funkcie

    Parameters
    ----------
    yvalues : TYPE
        DESCRIPTION.
    frac_harmonics : TYPE, optional
        DESCRIPTION. The default is 1.0.
    deg_polyfit : TYPE, optional
        DESCRIPTION. The default is 2.
    extrapolate_with : TYPE, optional
        DESCRIPTION. The default is 0.
    fraction_signal : TYPE, optional
        DESCRIPTION. The default is 1.0.
    mph : TYPE, optional
        DESCRIPTION. The default is 0.4.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    # N_original = len(yvalues)
    fft_x_orig, fft_y_orig, p2_orig = construct_fft(yvalues, deg_polyfit, real_abs_only=False)
    xvalues, yvalues, fft_x_r, fft_y_r, peaks = get_integer_no_of_periods(yvalues, fft_x_orig, fft_y_orig,
                                                                          frac=fraction_signal, mph=mph)
    fft_x, fft_y, p2 = construct_fft(yvalues, deg_polyfit, real_abs_only=False)

    N = len(yvalues)

    xvalues_full = np.arange(0, N + extrapolate_with)

    # inverzne spracovanie casoveho radu za predpokladu vyberu najvyznamnejsich amplitud
    restored_sig, idx, noHarm = restore_signal_from_fft(fft_x, fft_y, N, extrapolate_with, frac_harmonics)
    restored_sig = restored_sig + p2(xvalues_full)

    return restored_sig[-extrapolate_with:], fft_x_orig, fft_y_orig, fft_x_r, fft_y_r, fft_x, fft_y, xvalues,\
        yvalues, peaks, idx, noHarm, p2
