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

    def __init__(self, decoder):
        
        # pripravim si dataframe dat, na ktorych chcem vykreslit.
        d = decoder.getDFfull()
        
        x = d["TOTAL NB"]
        
        #y = d["Dub"]
        #y = d["Bezměrov"]
        #y = d["Sviňomazy"]
        y = d["MS Bečov"]
        
        # vykreslenie linearneho vztahu
        charts.scatterPlot(x, y)
        
        # ulozenie do adresara: MELTODO bude do configu upresnena adresa a 
        # nazov adresara
        
