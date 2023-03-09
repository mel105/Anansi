#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:02:49 2023

@author: mel
"""

import pandas as pd


def decoder(fileName, filePath, confObj):
    """


    Parameters
    ----------
    fileName : TYPE
        DESCRIPTION.
    filePath : TYPE
        DESCRIPTION.
    confObj : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """

    df = pd.read_excel(filePath+"/"+fileName[0]+".xlsx")
    df.rename(columns={df.columns[0]: "DATE"}, inplace=True)

    # Filtrovanie dat podla zadanych limitnych datumov
    df = df[(df['DATE'] >= confObj.getBeg()) & (df['DATE'] <= confObj.getEnd())]

    return df
