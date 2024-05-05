#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 13:56:02 2023

@author: mel
"""

import numpy as np
import datetime as dat
import pandas as pd


def forecast(time_data, ssa_data, n_future, indexes, noHarm, fft_x, fft_y, trd):

    n_fit = len(ssa_data)
    n = n_fit+n_future
    frc_time = np.arange(0, n)
    frc_signal = np.zeros(n)

    # 1. forecast pomocou harmonickej funkcie
    for i in indexes[:1 + noHarm * 2]:

        ampli = np.absolute(fft_y[i]) / n_fit
        phase = np.angle(fft_y[i])

        frc_signal += ampli * np.cos(2 * np.pi * fft_x[i] * frc_time + phase)

    frc_signal = frc_signal + trd(frc_time)

    # 2. uprava casoveho vektoru tak, aby som vyrobil timedate
    time_data = pd.to_datetime(time_data)
    time_data_ex = time_data.copy()
    mBeg = time_data.iloc[-1]
    mEnd = mBeg + dat.timedelta(n_future-1)
    ftr = pd.date_range(start=mBeg, end=mEnd, freq="24h")
    time_data_ex = time_data_ex.append(ftr.to_series())
    time_data_ex = time_data_ex.reset_index()
    time_data_ex = time_data_ex.drop(["index"], axis=1)
    time_data_ex = time_data_ex.rename(columns={0: "Time"})
    time_data_ex["Forecast"] = frc_signal

    # 3. grupovanie dat na rocne a mesacne sumy
    time_data_ex["YM"] = [i.strftime("%Y-%m") for i in list(time_data_ex.Time)]
    time_data_ex["YEAR"] = [i.strftime("%Y") for i in list(time_data_ex.Time)]
    time_data_ex["MONTH"] = [i.strftime("%m") for i in list(time_data_ex.Time)]

    # Rocne suhrny
    sumsOverYears = time_data_ex.groupby(["YEAR"]).sum(numeric_only=True)

    # Mesacne suhrny
    sumsOverMonths = time_data_ex.groupby(["YM", "MONTH"]).sum(numeric_only=True)

    return sumsOverYears, sumsOverMonths, time_data_ex
