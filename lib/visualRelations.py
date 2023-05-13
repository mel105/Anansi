#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:56:58 2023

@author: mel

Trieda obsahuje postup vyenerovania scatter plotov, ktore zobrazuju linearitu
medzi skumanym parametrom a nezavyslymmi parametrami
"""

import lib.charts as charts


class visualRelations:

    def __init__(self, config, decoder):

        # pripravim si dataframe dat, na ktorych chcem vykreslit.
        d = decoder.getDFfull()

        ufg = d["TOTAL NB"]
        tm = d["DATE"]
        saveFigPath = config.getOutLocalPath()+"/"+config.getFigFolderName()

        # ostranenie stlpcov, ktore nechcem, aby som medzi sebou porovnaval
        d = d.drop(columns=["DATE", "TOTAL NB"])

        # for cyklus cez vsetky stanice
        for (station, flow) in d.iteritems():

            print("Generating plots for: ", station, " station")

            # station = "Dub"
            mTitle = f"UFG vs FLOW comparison at the <b>{station}</b> station"

            # flow = d[station]

            # vykreslenie na y ose tok a na y2 ose celkove straty
            fig = charts.lineBarPlot(tm, ufg, flow, mTitle, "DATE", "UFG [kWh]", "FLOW [kWh]")

            fig.write_image(file=saveFigPath+"/"+station+"_flow_vs_ufg_A.png", format="png")

            # vykreslenie linearneho vztahu
            fig = charts.scatterPlot(ufg, flow, mTitle, "UFG [kWh]", "FLOW [kWh]")

            fig.write_image(file=saveFigPath+"/"+station+"_flow_vs_ufg_B.png", format="png")
