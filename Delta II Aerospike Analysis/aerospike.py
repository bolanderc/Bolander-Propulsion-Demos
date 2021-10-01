#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 10:20:24 2021

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt
import

class Aerospike:
    def __init__(self):
        self.mu = 3.9860e5  # Geocentric gravitational constant, km^3/s^2
        self.R_e = 6371  # km
        self.h_orbit = 860.06  # km
        self.R_e_ksc = 6373.1857  # km
        self.theta_orbit = np.deg2rad(28.7)  # rad

    def delta_v_req(self):
        V_orbit = np.sqrt(self.mu/(self.R_e + self.h_orbit))
        h_0 = 0.
        Omega_e = 2.*np.pi/((23 + 56/60 + 4.0905/3600)*3600)
        V_rot = (self.R_e_ksc + h_0)*Omega_e*np.cos(self.theta_orbit)
        print(V_rot)
        self.DV_kinematic = V_orbit - V_rot
        self.DV_potential = np.sqrt(2.*self.mu*self.h_orbit/((self.R_e + self.h_orbit)**2))
        self.DV_req = np.sqrt(self.DV_kinematic**2 + self.DV_potential**2)

    def delta_v_available(self):
        pass

    def _delta_v_stage1(self):
        pass

    def _stage1_exit_conds(self):


case1 = Aerospike()
case1.delta_v_req()
