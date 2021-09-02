#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 09:46:22 2021

@author: christian
"""

import pytest
import moc
import numpy as np
import os.path

def test_prandtl_meyer():
    nu = np.deg2rad(0.1257)
    gamma = 1.4
    M0 = 1.2
    M = moc.prandtl_meyer(nu, gamma, M0)
    test1 = (M - 1.020) <= 1e-4

    nu = np.deg2rad(25.83)
    M0 = 2.0
    M = moc.prandtl_meyer(nu, gamma, M0)
    test2 = (np.round(M, 4) - 1.9800) <= 1e-4
    assert test1*test2
