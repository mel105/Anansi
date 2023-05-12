#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:56:58 2023

@author: mel

Trieda obsahuje postup vyenerovania scatter plotov, ktore zobrazuju linearitu
medzi skumanym parametrom a nezavyslymmi parametrami
"""

import lib.charts as charts


class linearity:

    def __init__(self, config, decoder):

        # pripravim si dataframe dat, na ktorych chcem vykreslit.
        d = decoder.getDFfull()

        ufg = d["TOTAL NB"]
        tm = d["DATE"]
        saveFigPath = config.getOutLocalPath()+"/"+config.getFigFolderName()

        # for cyklus cez vsetky stanice

        station = "Dub"
        mTitle = f"Porovnanie UFG vs. FLOW na stanici <b>{station}</b>"

        flow = d[station]

        # vykreslenie na y ose tok a na y2 ose celkove straty
        fig = charts.lineBarPlot(tm, ufg, flow, mTitle, "DATE", "UFG [kWh]",
                                 "FLOW [kWh]")

        fig.write_image(file=saveFigPath+"/"+station+".png", format=".png")

        # vykreslenie linearneho vztahu
        # charts.scatterPlot(ufg, flow)

        # ulozenie do adresara: MELTODO bude do configu upresnena adresa a
        # nazov adresara
