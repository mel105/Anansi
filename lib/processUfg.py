#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:12:33 2023

@author: mel
"""

import numpy as np


class processUfg:
    """
    Trieda ma za ulohu:
        1. odvodit relativne ufg, t.j. percento straty k input toku. Trieda dedi triedu processModel,
        ktora sa postara o to, aby sa spocitali modelovane relativne ufg.
    """

    def __init__(self, prepinac, modelObj):

        self._relRealUfG = modelObj.getRealUfG() / modelObj.getIntakes().reshape((-1, 1))

        if prepinac:

            self._relCalcUfG = np.array(modelObj.getAltModel()).reshape(
                (-1, 1)) / modelObj.getIntakes().reshape((-1, 1))
        else:

            self._relCalcUfG = modelObj.getModel().reshape((-1, 1)) / modelObj.getIntakes().reshape((-1, 1))

    def getRealUfG(self):
        """
        Funkcia vrati realne relativne hodnoty UfG

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._relRealUfG

    def getCalcUfG(self):
        """
        Funkcia vrati modelovane relativne hodnoty UfG

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._relCalcUfG
