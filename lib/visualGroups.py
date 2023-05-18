#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 11:17:25 2023

@author: mel

Trieda sa venuje spracovaniu dat z pohladu grupovania podla mesiaca, tyzdna atp.
"""
import lib.charts as charts


class visualGroups:

    def __init__(self, config, decoder):

        # pripravim si data
        YE = decoder.getDFYearly()
        MO = decoder.getDFMonthly()
        WE = decoder.getDFWeekly()

        # vyrobenie grafu: Casove rady po staniciach
        # self._visualStations(YE)
        # self._visualStations(MO)
        # self._visualStations(WE)

        # heat mapy, na ktorych zobrazujem tyzdenne, mesacne a rocne priemery.

    def _visualStations(self, df):

        # funkcia zobrazi multilinechart po jednotlivych staniiciach

        if df.columns.str.contains("YEAR").any():
            df = df.rename(columns={"YEAR": "DATE"})

        elif df.columns.str.contains("YM").any():
            df = df.rename(columns={"YM": "DATE"})
            df = df.drop(["MONTH"], axis=1)

        elif df.columns.str.contains("YW").any():
            df = df.rename(columns={"YW": "DATE"})
            df = df.drop(["WEEK"], axis=1)

        mtit = "Presentation of monthly averages at each station"
        xLab = "DATE"
        yLab = "Monthly Averages [kWh]"

        fig = charts.multilineChart(df, xLab, yLab, mtit)

        return fig
