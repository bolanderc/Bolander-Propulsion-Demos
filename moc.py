#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 09:23:34 2021

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt


def prandtl_meyer(nu, gamma, M):
    error = 100.
    while error > 1e-12:
        nu_M = _nu_M(M, gamma)
        dnu_dM = _dnu_dM(M, gamma)
        Mp1 = M + (nu - nu_M)/dnu_dM
        error = np.abs((nu - nu_M)/nu)
        M = Mp1
    return M

def _nu_M(M, gamma):
    gp1 = gamma + 1.
    gm1 = gamma - 1.
    glau_coeff = M*M - 1.
    C1 = np.sqrt(gp1/gm1)
    C2 = np.arctan2(np.sqrt(gm1*glau_coeff), np.sqrt(gp1))
    C3 = np.arctan2(np.sqrt(glau_coeff), 1.)
    nu_M = C1*C2 - C3
    return nu_M

def _dnu_dM(M, gamma):
    glau_coeff = np.sqrt(M*M - 1.)
    gm1 = gamma - 1.
    C1 = 1./M
    C2 = glau_coeff
    C3 = 1. + gm1*M*M/2.
    dnu_dM = C1*C2/C3
    return dnu_dM

class MethodOfCharacteristics:
    def __init__(self):
        self.gamma = 1.4

    def _compat_constant(self, M, theta, left=True):
        nu = _nu_M(M, self.gamma)
        if not left:
            K = theta + nu
        else:
            K = theta - nu
        return K

    def _point3_props(self, K1, K2):
        theta_3 = (K1 + K2)/2.
        nu_3 = (K1 - K2)/2.
        M_3 = prandtl_meyer(nu_3, self.gamma, 1.5)
        return theta_3, nu_3, M_3

    def _mach_angle(self, M):
        mu = []
        for m in M:
            mu.append(np.arcsin(1./m))
        return mu

    def _char_slope(self, theta, mu):
        m_cm = 0.5*((theta[0] - mu[0]) + (theta[2] - mu[2]))
        m_cp = 0.5*((theta[1] + mu[1]) + (theta[2] + mu[2]))
        return m_cm, m_cp

    def _point3_xy(self, m_cm, m_cp, x, y):
        x1, x2 = x
        y1, y2 = y
        t_mcm = np.tan(m_cm)
        t_mcp = np.tan(m_cp)
        x3 = (x1*t_mcm - x2*t_mcp + y2 - y1)/(t_mcm - t_mcp)
        y3 = (t_mcm*t_mcp*(x1 - x2) + t_mcm*y2 - t_mcp*y1)/(t_mcm - t_mcp)
        return x3, y3

    def _unit_interal(self, M, theta, xy1, xy2):
        M1, M2 = M
        theta1, theta2 = theta
        x1, y1 = xy1
        x2, y2 = xy2
        Km1 = self._compat_constant(M1, theta1, left=False)
        Kp2 = self._compat_constant(M2, theta2, left=True)
        theta3, nu3, M3 = self._point3_props(Km1, Kp2)
        mu_i = self._mach_angle([M1, M2, M3])
        m_cm, m_cp = self._char_slope([theta1, theta2, theta3], mu_i)
        x3, y3 = self._point3_xy(m_cm, m_cp, [x1, x2], [y1, y2])
        return x3, y3, M3, theta3

case = MethodOfCharacteristics()
M = [2.0, 1.75]
theta = np.deg2rad([10., 5.])
xy1 = [1., 2.]
xy2 = [1.5, 1.]
x3, y3, M3, theta3 = case._unit_interal(M, theta, xy1, xy2)
print(x3, y3, M3, np.rad2deg(theta3))
