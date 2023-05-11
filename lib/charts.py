#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 19:43:10 2023

@author: mel
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# from scipy.stats import norm
import plotly.figure_factory as ff
# from scipy import stats
from statsmodels.graphics.gofplots import qqplot
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib
matplotlib.use("WebAgg")
pio.renderers.default = "browser"

def scatterPlot(xVec, yVec, mTitle=" ", xLabel=" ", yLabel=" "):
    """
    Metoda vykresli scatter plot vektorov x, y

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    mTitle : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig = go.Figure(data=go.Scattergl(
    x = xVec,
    y = yVec,
    mode='markers',
    marker=dict(
        color=10,
        colorscale='Viridis',
        line_width=1
        )
    ))

    fig.show()

def prediction(mtim, mdat, ptim, real, pred, ci):
    """
    Funkcia kresli predickciu na zaklade zvoleneho modelu a znamych buducich tokov

    Parameters
    ----------
    mtim : TYPE
        DESCRIPTION. Vektor casu validny pre model
    mdat : TYPE
        DESCRIPTION. Vektor dat pre model
    ptim : TYPE
        DESCRIPTION. Vektor casu validny pre hodnoty predikcie
    real : TYPE
        DESCRIPTION. Vektor realnych merani v casti predikcie
    pred : TYPE
        DESCRIPTION. Vektor odhadu v casti predikcie
    ci : TYPE
        DESCRIPTION. 95% interval spolahlivosti

    Returns
    -------
    int
        DESCRIPTION

    """

    fig, ax = plt.subplots(figsize=(15, 7))
    # zobrazenie modelu
    ax.plot(mtim, mdat, color="#b35151", label="Model Estimations")

    # zobrazenie predikovanej casti
    ax.plot(ptim, real, color="#e68484", label="Measurements")
    ax.plot(ptim, pred, color='#707070', label='Predictions')

    # ax.scatter(valid.index, sarima_preds)
    ax.fill_between(ptim, (pred-ci), (pred+ci), color='b', alpha=.1, label="95% Confidence Interval")
    ax.set_title("Predictions model")
    ax.set_xlabel('Date')
    ax.set_ylabel('UfG')
    ax.legend()

    plt.show()

    return 0


def heatmapDiagram(df):
    """
    Funkcia vrati tzv. heatmap, ktora zobrazuje vysledok korelacnej matice, cize zobrazuje zavislosti medzi
    parametrami

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """

    corrmat = df.drop(columns="DATE").corr()

    # #1
    fig1 = plt.figure(figsize=(50, 40))
    # define the mask to set the values in the upper triangle to True
    mask = np.triu(np.ones_like(corrmat, dtype=np.bool))
    heatmap = sns.heatmap(corrmat, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')

    heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize': 35}, pad=10)

    plt.savefig('my_plot_a.png')

    # #2
    fig2 = plt.figure(figsize=(25, 35))
    heatmap = sns.heatmap(corrmat[['TOTAL NB']].sort_values(by='TOTAL NB', ascending=False), vmin=-1, vmax=1,
                          annot=True, cmap='BrBG')
    heatmap.set_title("Features Correlating with TOTAL NB", fontdict={'fontsize': 35}, pad=10)

    plt.savefig('my_plot_2.png')
    # fig = go.Figure(data=go.Heatmap(corrmat, hoverongaps=False))

    """
    print('<HTML><HEAD><TITLE>Python Matplotlib Graph</TITLE></HEAD>')
    print('<BODY>')
    print('<CENTER>')
    print('<br><br>')
    print('<H3>Graph</H3>')
    print(mpld3.fig_to_html(fig1, d3_url=None, mpld3_url=None, no_extras=False, template_type='general',
                            figid=None, use_http=False))
    print('<br>')

    print('</CENTER>')
    print('</BODY>')
    print('</html>')
    """
    fig1.show()
    fig2.show()

    return fig1, fig2, corrmat, mask


def scatterMatrix(df):
    """
    Funkcia vrati scatter diagram matrix

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    fig = ff.create_scatterplotmatrix(df, diag='histogram', index='DATE',
                                      colormap=['rgb(100, 150, 255)', '#F0963C', 'rgb(51, 255, 153)'],
                                      colormap_type='seq', height=800, width=800)
    # fig = sns.pairplot(df, corner=True, diag_kind="kde")

    # fig = sns_plot.get_figure()

    fig.show()

    return fig


def histChart(res, title):
    """
    Funkcia vykresli histogram

    Parameters
    ----------
    res : TYPE numpy.array
        DESCRIPTION. Vektor, ktory obsahuje rozdiel medzi modelom a meraniami

    Returns
    -------
    None.



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
    """

    """
    x = res
    hist_data = [x]
    group_labels = ['distplot']  # name of the dataset

    # mean = np.mean(x)
    # stdev_pluss = np.std(x)
    # stdev_minus = np.std(x)*-1

    fig = ff.create_distplot(hist_data, group_labels, curve_type='kde')
    fig.update_layout(template='plotly_dark')

    fig2 = ff.create_distplot(hist_data, group_labels, curve_type='normal')
    normal_x = fig2.data[1]['x']
    normal_y = fig2.data[1]['y']

    fig.add_traces(go.Scatter(x=normal_x, y=normal_y, mode='lines',
                              line=dict(color='rgba(0,255,0, 0.6)',
                                        width=1),
                              name='normal'
                                  ))
    """

    fig = go.Figure(data=[go.Histogram(x=res, marker=go.histogram.Marker(color="orange"))])
    fig.update_layout(

        title=dict(text=title),
        xaxis_title="Residuals",
        yaxis_title="Frequency",

        autosize=False,
        width=1000,
        height=500,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title="Resituals"
        ),

        showlegend=True,
        plot_bgcolor='white'
    )
    fig.show()

    return fig


def qqChart(data, title):
    """
    Funkcia vrati tzv. QQ plot, ktory poukazuje na normalitu dat, ak sa qvantily dat sustredia okolo ich
    teoretickych hodnot

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    """

    qqplot_data = qqplot(data, line='s').gca().lines

    fig = go.Figure()

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[0].get_xdata(),
        'y': qqplot_data[0].get_ydata(),
        'mode': 'markers',
        'marker': {
            'color': '#19d3f3'
        }
    })

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[1].get_xdata(),
        'y': qqplot_data[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': '#636efa'
        }

    })

    fig['layout'].update({
        'title': 'Quantile-Quantile Plot',
        'xaxis': {
            'title': 'Theoritical Quantities',
            'zeroline': False
        },
        'yaxis': {
            'title': 'Sample Quantities'
        },
        'showlegend': False,
        'width': 800,
        'height': 700,
    })

    fig.show()

    return fig


def boxChart(data, title, boxp="all"):
    """
    Funkcia vrati objekt triedy plotly a obshauje data k vyresleniu box plotu

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """

    fig = go.Figure(data=[go.Box(y=data,
                                 boxpoints=boxp,
                                 jitter=0.3,
                                 pointpos=-1.8,
                                 name=title,
                                 )])

    fig.show()

    return fig


def multilineChart(df, xLabel=" ", yLabel=" ", title=" "):
    """
    Funkcia vrati multiline obrazok

    Parameters
    ----------
    dates : TYPE
        DESCRIPTION.
    vals : TYPE
        DESCRIPTION.
    xLabel : TYPE, optional
        DESCRIPTION. The default is " ".
    yLabel : TYPE, optional
        DESCRIPTION. The default is " ".
    title : TYPE, optional
        DESCRIPTION. The default is " ".

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """

    fig = go.Figure()

    _, b = df.shape

    for i in range(1, b):
        fig.add_trace(go.Scatter(x=df["DATE"], y=df[df.columns[i]], mode='lines', name=df.columns[i],
                                 line=dict(width=2), connectgaps=True, ))

    fig.update_layout(

        title=title,
        xaxis_title=xLabel,
        yaxis_title=yLabel,

        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
        ),
        autosize=False,
        width=1000,
        height=500,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),

        showlegend=True,
        plot_bgcolor='white'
    )

    fig.show()
    return fig


def lineChart(dates, vals, xLabel=" ", yLabel=" ", title=" "):
    """
    Funkcia vrati objekt typu plotly a vysledkom funkcie je tzv. line chart

    Parameters
    ----------
    dates : TYPE
        DESCRIPTION.
    vals : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """

    colors = ['rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)']

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dates, y=vals, mode='lines', line=dict(color=colors[0], width=2),
                             connectgaps=True, ))

    fig.update_layout(

        title=title,
        xaxis_title=xLabel,
        yaxis_title=yLabel,

        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
        ),
        autosize=False,
        margin=dict(
            autoexpand=True,
            l=50,
            r=10,
            t=55,
        ),
        showlegend=False,
        plot_bgcolor='white'
    )

    fig.show()

    return fig
