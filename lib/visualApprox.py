#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 13:35:28 2023

@author: mel
"""

import numpy as np
import plotly.graph_objects as go
from scipy.signal import welch


def plot_data(time_data, ssa_data):

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

    return fo1, yvalues_detrended, yvalues_trend


def plot_fft(yvalues_detrended, fft_x_r, fft_y_r, peaks):

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

        fo2.add_trace(go.Scatter(x=xl_list, y=yl_list, mode="text", name=" ", text=text_list,
                                 textposition="top center"))
        # fo2.show()

    return fo2


def plot_approx(time_data, ssa_data, sig):

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
    return fo3


def plot_forecast(time_data_ex, ssa_data):
    # Zobrazenie originalnej rady a forecastu do buducnost
    fo4 = go.Figure()
    fo4.add_trace(go.Scatter(x=time_data_ex.Time, y=ssa_data, name="Analysed time series", mode="lines",
                             line=dict(color="royalblue", width=3)))
    fo4.add_trace(go.Scatter(x=time_data_ex.Time, y=time_data_ex.Forecast, name="Required period Forecast",
                             mode="lines", line=dict(color="tomato", width=3)))
    fo4.update_layout(title="Presentation of forecasted values of analysed parameter",
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
    # fo4.show()

    return fo4
