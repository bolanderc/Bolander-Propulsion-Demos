#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 19:36:40 2021

@author: christian
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/home/christian/Python Projects/School Work/Fall 2019/MAE 5420')
import KobayashiMaru
comp_codes = KobayashiMaru.KobayashiMaru()


class Loon:
    def __init__(self):
        self.F_T_sl = 2200.  # N
        self.V_inf = 180.  # m/s
        self.A_inlet = 0.145  # m^2
        self.Q_R = 40.0e6  # J/kg
        self.eta_comb = 0.9
        self.t_burn = 1800.  # s
        self.T_exit = 735.  # K
        self.cp_inlet = 1.005e3  # J/kg-K
        self.cp_exit = 1.12e3  # J/kg-K
        self.W_launch = 2150.  # kg
        self.rho_sl = 1.225  # kg/m^3
        self.R_u = 8314.4612  # J/kg-K
        self.MW_inlet = 32.  # kg/kg-mol
        self.Rg_inlet = 287.056  # J/kg-K
        self.gamma_inlet = 1.4
        self.p_inf = 101.325e3  # kPa
        self.T_inf = 288.15  # K
        self.g_0 = 9.81  # m/s^2

    def _massflowrate(self):
        mdot = self.rho_sl*self.V_inf*self.A_inlet
        return mdot

    def _exit_velocity(self):
        # Assume mdot_in >> mdot_fuel
        V_exit = self.F_T_sl/self.mdot_inlet + self.V_inf
        return V_exit

    def _max_temp_in_engine(self):
        T_comb = (self.cp_exit*self.T_exit + self.V_exit**2/2.)/self.cp_exit
        return T_comb

    def _air_to_fuel(self):
        num = self.cp_exit*self.T_exit + 0.5*self.V_exit**2 - self.h_fuel
        denom = self.cp_inlet*self.T_inf + 0.5*self.V_inf**2 - self.cp_exit*self.T_exit - 0.5*self.V_exit**2
        f = num/denom
        return f

    def _tsfc(self):
        TSFC = self.mdot_fuel/self.F_T_sl
        return TSFC

    def _l_over_d(self):
        num = self.R_avg
        denom = (self.Isp*self.V_inf*np.log(self.W_launch/self.W_final))
        L_D = num/denom
        return L_D

    def _range(self):
        R_avg = self.V_inf*self.t_burn
        return R_avg

    def _Isp(self):
        Isp = self.F_T_sl/(self.g_0*self.mdot_fuel)
        return Isp

    def _m_final(self):
        M_init = self.W_launch
        M_burn = self.mdot_fuel*self.t_burn
        return M_init - M_burn


    def analysis(self):
        self.h_fuel = self.eta_comb*self.Q_R  # J/kg
        self.mdot_inlet = self._massflowrate()
        print(f"The air mass flow rate into the engine is: {self.mdot_inlet:1.4f} kg/s")
        self.V_exit = self._exit_velocity()
        print(f"The exhaust velocity is: {self.V_exit:1.4f} m/s")
        self.T_comb = self._max_temp_in_engine()
        print(f"The maximum temperature in the engine (T_comb) is: {self.T_comb:1.4f} K")
        self.M_exit = self.V_inf/np.sqrt(self.gamma_inlet*self.Rg_inlet*self.T_inf)
        self.P_comb = comp_codes.isenflowprops("P", self.M_exit,
                                               gamma=self.gamma_inlet,
                                               flow=self.p_inf)
        print(f"The maximum pressure in the engine (P_0inf) is: {self.P_comb/1000.:1.4f} kPa")
        self.f = self._air_to_fuel()
        self.mdot_fuel = self.mdot_inlet/self.f
        self.TSFC = self._tsfc()*2.204*3600*4.4495  # lbm/lbf-hr
        print(f"The thrust-specific fuel consumption is : {self.TSFC:1.4f} lbm/(lbf-hr)")
        self.R_avg = self._range()
        print(f"The average range is : {self.R_avg/1000.:1.4f} km")
        self.Isp = self._Isp()
        self.W_final = self._m_final()
        self.L_D_mean = self._l_over_d()
        print(f"The mean lift-to-drag ratio is : {self.L_D_mean:1.4f}")




if __name__ == "__main__":
    plt.close('all')
    case = Loon()
    case.analysis()

