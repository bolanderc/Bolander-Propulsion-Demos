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

def test_A_Astar():
    A_Astar = 77.5
    M0 = 3.0
    gamma = 1.25
    M = moc.A_Astar(A_Astar, gamma, M0)
    test = (M - 5.09747) <= 1e-5
    assert test

def test_internal_unit():
    case = moc.MethodOfCharacteristics()
    M = [2.0, 1.75]
    theta = np.deg2rad([10., 5.])
    xy1 = [1., 2.]
    xy2 = [1.5, 1.]
    x3, y3, M3, theta3 = case._unit_internal(M, theta, xy1, xy2)
    result = np.round([x3, y3, M3, theta3], 5)
    answer = np.round([2.17091, 1.57856, 1.96198, np.deg2rad(1.57856)], 5)
    test = [abs(i - j) for i,j in zip(result, answer)]
    assert test
