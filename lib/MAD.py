#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Now  1 10:51:36 2023

@author: mel

Skript obsahuje implemntaciu metody MAD (median absolute deviation) pre ucel identifkacie
outliers.
"""

import statistics as stat
import plotly.graph_objects as go


def MAD(data):

    # Outliers replacing by median
    med1 = stat.median(data)

    MAD = stat.median(abs(data-med1))

    # Tukey's fence
    low = med1 - 4.45*MAD
    upp = med1 + 4.45*MAD

    # outlier indexes
    uppidx = data >= upp
    lowidx = data <= low

    # outliers replacement

    # data.loc[lowidx, "HOM"] = med1
    # data.loc[uppidx, "HOM"] = med1

    return med1, MAD, lowidx, uppidx


def plot(data_X, data_Y, lowidx, uppidx):

    fig_out = go.Figure()
    fig_out.add_trace(go.Scatter(x=data_X, y=data_Y, mode="lines"))
    fig_out.add_trace(go.Scatter(x=data_X[lowidx], y=data_Y[lowidx], mode="markers"))
    fig_out.add_trace(go.Scatter(x=data_X[uppidx], y=data_Y[uppidx], mode="markers"))
    fig_out.update_layout(
        title="Presentation the Outliers in Analysed Time Series",
        xaxis_title="TIME",
        yaxis_title="UfG [kWh]",
        autosize=False,
        width=800,
        height=400,

        # yaxis=dict(
        #     autorange=True,
        #     showgrid=True,
        #     zeroline=True,
        #     dtick=250,
        #     gridcolor="rgb(255, 255, 255)",
        #     gridwidth=1,
        #     zerolinecolor="rgb(255, 255, 255)",
        #     zerolinewidth=2,
        # ),
        margin=dict(l=40, r=30, b=80, t=100,),
        paper_bgcolor="rgb(243, 243, 243)",
        plot_bgcolor="rgb(243, 243, 243)",
        showlegend=False,
    )

    return fig_out


def replacement(data, lowidx, uppidx, med1):

    # replace the outlier values by the median ones
    data.loc[lowidx] = med1
    data.loc[uppidx] = med1

    return data
