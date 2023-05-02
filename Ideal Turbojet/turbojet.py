#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:58:31 2021

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from prettytable import PrettyTable
sys.path.insert(1, '/home/christian/Python Projects/School Work/Fall 2019/MAE 5420')
import KobayashiMaru
comp_codes = KobayashiMaru.KobayashiMaru()
import scipy.optimize as optimize
# from bokeh.plotting import figure
# from bokeh.resources import CDN
# from bokeh.embed import file_html
import matplotlib.backends.backend_pdf

class brayton_turbojet:
    def __init__(self):
        self.MW = 28.96443  # kg/kg-mol
        self.gamma = 1.40
        self.R_g = 287.058  # J/kg-K
        self.T_inf = 230.  # K
        self.p_inf = 26.0e3  # Pa
        self.V_inf = 220.  # m/s
        self.M_inf = self.V_inf/np.sqrt(self.gamma*self.R_g*self.T_inf)
        self.rho_inf = self.p_inf/(self.R_g*self.T_inf)
        self.R_u = 8314.4612  # J/kg-K
        self.c_p = self.gamma*self.R_g/(self.gamma - 1)  # J/kg-K
        self.c_v = self.c_p/self.gamma  # J/kg-K
        self.eta_diff = 1.0
        self.eta_comp = 0.85
        self.eta_turb = 0.9
        self.T_3 = 1400  # K
        self.compression_ratio = 11.
        self.eta_nozz = 1.0
        self.p_5 = self.p_inf  # Pa
        self.table_data = PrettyTable(['Station', 'p, kPa', 'T, K',
                                       r'h0, kJ/kg', r's, kJ/(kg-K)',
                                       r'rho, kg/m^3', r'v, m^3/kg'], float_format='1.4')

    def _entropy_change(self, T_ratio, p_ratio):
        delta_s = self.c_p*np.log(T_ratio) - self.R_g*np.log(p_ratio)
        return delta_s


    def _diffuser(self):
        self.s_inf = 0.
        D_in = 60.96e-2  # m
        D_out = 1.5*D_in  #m
        self.T_0inf = comp_codes.isenflowprops("T", self.M_inf, gamma=self.gamma, flow=self.T_inf)
        self.T_01 = self.T_0inf
        self.h_0inf = self.c_p*self.T_0inf
        self.h_01 = self.h_0inf
        self.h_inf = self.h_01 - self.V_inf**2/2.
        self.P_0inf = comp_codes.isenflowprops("P", self.M_inf, gamma=self.gamma, flow=self.p_inf)
        self.P_01 = self.P_0inf
        self.rho_0inf = self.P_0inf/(self.R_g*self.T_0inf)
        A_in = np.pi*D_in**2/4.
        A_out = np.pi*D_out**2/4.
        A_Astar = comp_codes.area_mach(self.M_inf, gamma=1.4)
        self.Astar_diff = A_in/A_Astar
        self.mdot_diff = (self.p_inf/(self.R_g*self.T_inf))*self.V_inf*A_in
        self.M_1 = optimize.minimize(comp_codes.area_mach, 0.5*self.M_inf, args=(A_out/self.Astar_diff, self.gamma)).x[0]
        self.V_1 = self.M_1*np.sqrt(self.gamma*self.R_g*self.T_inf)
        self.T_1 = comp_codes.isenflowprops("T", self.M_1, gamma=self.gamma, stagnation=self.T_01)
        self.p_1 = comp_codes.isenflowprops("P", self.M_1, gamma=self.gamma, stagnation=self.P_01)
        self.h_1 = self.h_01 - self.V_1**2/2.
        self.rho_1 = self.p_1/(self.R_g*self.T_1)
        self.rho_01 = self.P_01/(self.R_g*self.T_01)
        self.s_1 = self.s_inf
        self.table_data.add_row(['inf', self.p_inf/1000., self.T_inf, self.h_0inf/1000.,
                                 self.s_inf/1000., self.rho_inf, 1./self.rho_inf])
        self.table_data.add_row(['1', self.p_1/1000., self.T_1, self.h_01/1000.,
                                 self.s_1/1000., self.rho_1, 1./self.rho_1])

    def _compressor(self):
        self.M_2 = 0.
        self.V_2 = 0.
        self.P_02 = self.compression_ratio*self.P_01
        self.p_2 = self.compression_ratio*self.p_1
        self.T_02ideal = self.T_01*(self.P_02/self.P_01)**((self.gamma - 1.)/self.gamma)
        self.h_02ideal = self.c_p*self.T_02ideal
        self.h_02 = (self.h_02ideal - self.h_01)/self.eta_comp + self.h_01
        self.h_2 = self.h_02 - self.V_2**2/2.
        self.T_02 = self.h_02/self.c_p
        self.s_2 = self._entropy_change(self.T_02/self.T_01, self.P_02/self.P_01) + self.s_1
        self.T_2 = self.h_2/self.c_p
        self.rho_2 = self.p_2/(self.R_g*self.T_2)
        self.rho_02 = self.P_02/(self.R_g*self.T_02)
        self.work_compressor = self.h_02 - self.h_01
        self.table_data.add_row(['2', self.p_2/1000., self.T_2, self.h_02/1000.,
                                 self.s_2/1000., self.rho_2, 1./self.rho_2])

    def _combustor(self):
        self.p_3 = self.p_2
        self.P_03 = self.P_02
        self.M_3 = self.M_2
        self.V_3 = self.V_2
        self.s_3 = self.c_p*np.log(self.T_3/self.T_2) + self.s_2
        self.T_03 = self.T_02*np.exp((self.s_3 - self.s_2)/self.c_p)
        self.Q_comb = self.c_p*(self.T_03 - self.T_02)
        self.h_3 = self.h_2 + self.Q_comb + self.V_2**2/2. - self.V_3**2/2.
        self.h_03 = self.h_3 + self.V_3**2/2.
        self.rho_3 = self.p_3/(self.R_g*self.T_3)
        self.rho_03 = self.P_03/(self.R_g*self.T_03)
        self.table_data.add_row(['3', self.p_3/1000., self.T_3, self.h_03/1000.,
                                 self.s_3/1000., self.rho_3, 1./self.rho_3])

    def _turbine(self):
        self.work_turbine = self.work_compressor
        self.h_04 = self.h_03 - self.work_turbine
        self.h_04ideal = self.h_03 - self.work_turbine/self.eta_turb
        self.T_04 = self.h_04/self.c_p
        self.T_04ideal = self.h_04ideal/self.c_p
        self.P_04 = self.P_03*(self.T_04ideal/self.T_03)**(self.gamma/(self.gamma - 1.))
        self.V_4 = np.sqrt(2.*(self.h_03 - self.work_turbine + self.V_3**2/2. - self.h_04))
        self.h_4 = self.h_04 - self.V_4**2/2.
        self.h_4ideal = self.h_04ideal - self.V_4**2/2.
        self.T_4 = self.h_4/self.c_p
        self.T_4ideal = self.h_4ideal/self.c_p
        self.p_4 = self.p_3*(self.T_4ideal/self.T_3)**(self.gamma/(self.gamma - 1.))
        self.M_4 = self.V_4*np.sqrt(self.gamma*self.R_g*self.T_4)
        self.rho_4 = self.p_4/(self.R_g*self.T_4)
        self.rho_04 = self.P_04/(self.R_g*self.T_04)
        self.s_4 = self._entropy_change(self.T_04/self.T_03, self.P_04/self.P_03) + self.s_3
        self.table_data.add_row(['4', self.p_4/1000., self.T_4, self.h_04/1000.,
                                 self.s_4/1000., self.rho_4, 1./self.rho_4])

    def _nozzle(self):
        self.p_5 = self.p_inf
        self.T_5 = self.T_4*(self.p_5/self.p_4)**((self.gamma - 1.)/self.gamma)
        self.h_5 = self.c_p*self.T_5
        self.V_5 = np.sqrt(2.*(self.h_4 - self.h_5))
        self.M_5 = self.V_5/np.sqrt(self.gamma*self.R_g*self.T_5)
        self.P_05 = comp_codes.isenflowprops("P", self.M_5, gamma=self.gamma, flow=self.p_5)
        self.T_05 = comp_codes.isenflowprops("T", self.M_5, gamma=self.gamma, flow=self.T_5)
        self.rho_5 = self.p_5/(self.R_g*self.T_5)
        self.rho_05 = self.P_05/(self.R_g*self.T_05)
        self.h_05 = self.h_5 + self.V_5**2/2.
        self.s_5 = self.s_4
        self.table_data.add_row(['5', self.p_5/1000., self.T_5, self.h_05/1000.,
                                 self.s_5/1000., self.rho_5, 1./self.rho_5])

    def output(self):
        self._diffuser()
        self._compressor()
        self._combustor()
        self._turbine()
        self._nozzle()
        output_text = ("---------------------------\n" +
                       " Ideal Turbojet Analysis \n" +
                       "---------------------------" +
                       "\n Properties at Each State in the Cycle\n")
        print(output_text)
        print(self.table_data)
        print(f"\nHeat Transfer Rate in the Combustor : {self.Q_comb/1000.:1.4f} kW")
        print(f"Velocity at Nozzle Exit : {self.V_5:1.4f} m/s")
        self.F_propulsive = self.mdot_diff*(self.V_5 - self.V_inf)
        print(f"Propulsive Force : {self.F_propulsive/1000.:1.4f} kN or {self.F_propulsive/1000.*224.809:1.4f} lbf")
        self.W_propulsive = self.F_propulsive*self.V_inf
        print(f"Propulsive Work : {self.W_propulsive/1000.:1.4f} kW")
        self.eta_propulsive = 2.*((self.V_inf/self.V_5) - (self.V_inf/self.V_5)**2)/(1. - (self.V_inf/self.V_5)**2)
        self.eta_thermal = (0.5*self.V_5**2)*(1. - (self.V_inf/self.V_5)**2)/(self.h_03 - self.h_02)
        self.KE_excess = 0.5*self.mdot_diff*self.V_5**2*(1. - self.V_inf/self.V_5)
        self.eta_total = self.eta_propulsive*self.eta_thermal
        print(f"Propulsive Efficiency : {self.eta_propulsive*100:1.4f} \%")
        print(f"Thermal Efficiency : {self.eta_thermal*100:1.4f} \%")
        print(f"Total Efficiency : {self.eta_total*100.:1.4f} \%")
        print(f"Excess Kinetic Energy : {self.KE_excess/1000.:1.4f} kJ")


    def plot_states(self, pdf):
        T_fig, T_axis = plt.subplots()
        stages = np.arange(0, 6, 1)
        T_stages = np.array([self.T_inf, self.T_1, self.T_2, self.T_3,
                             self.T_4, self.T_5])
        T0_stages = np.array([self.T_0inf, self.T_01, self.T_02, self.T_03,
                              self.T_04, self.T_05])
        T_axis.plot(stages, T_stages, color='k', label='Static Temperature, $T$')
        T_axis.plot(stages, T0_stages, color='k', linestyle=':', label='Stagnation Temperature, $T_0$')
        T_axis.set_xticks(stages)
        T_axis.set_xlabel('Stage')
        T_axis.set_ylabel(r'Temperature, K')
        T_axis.set_xlim(0, 5)
        T_axis.legend()

        P_fig, P_axis = plt.subplots()
        p_stages = np.array([self.p_inf, self.p_1, self.p_2, self.p_3,
                             self.p_4, self.p_5])
        P0_stages = np.array([self.P_0inf, self.P_01, self.P_02, self.P_03,
                              self.P_04, self.P_05])
        P_axis.plot(stages, p_stages/1000., color='k', label='Static Pressure, $p$')
        P_axis.plot(stages, P0_stages/1000., color='k', linestyle=':', label='Stagnation Pressure, $P_0$')
        P_axis.set_xticks(stages)
        P_axis.set_xlabel('Stage')
        P_axis.set_ylabel('Pressure, kPa')
        P_axis.set_xlim(0, 5)
        P_axis.legend()

        M_fig, M_axis = plt.subplots()
        M_stages = np.array([self.M_inf, self.M_1, self.M_2, self.M_3,
                             self.M_4, self.M_5])
        M_axis.plot(stages, M_stages, color='k')
        M_axis.set_xticks(stages)
        M_axis.set_xlabel('Stage')
        M_axis.set_ylabel('Mach Number')
        M_axis.set_xlim(0, 5)

        V_fig, V_axis = plt.subplots()
        V_stages = np.array([self.V_inf, self.V_1, self.V_2, self.V_3,
                             self.V_4, self.V_5])
        V_axis.plot(stages, V_stages, color='k')
        V_axis.set_xticks(stages)
        V_axis.set_xlabel('Stage')
        V_axis.set_ylabel('Velocity, m/s')
        V_axis.set_xlim(0, 5)

        s_fig, s_axis = plt.subplots()
        s_stages = np.array([self.s_inf, self.s_1, self.s_2, self.s_3,
                             self.s_4, self.s_5])
        s_axis.plot(stages, s_stages/1000., color='k')
        s_axis.set_xticks(stages)
        s_axis.set_xlabel('Stage')
        s_axis.set_ylabel('Specific Entropy, kJ/(kg-K)')
        s_axis.set_xlim(0, 5)

        h_fig, h_axis = plt.subplots()
        h_stages = np.array([self.h_inf, self.h_1, self.h_2, self.h_3,
                             self.h_4, self.h_5])
        h0_stages = np.array([self.h_0inf, self.h_01, self.h_02, self.h_03,
                              self.h_04, self.h_05])
        h_axis.plot(stages, h_stages/1000., color='k', label='Static Enthalpy, $h$')
        h_axis.plot(stages, h0_stages/1000., color='k', linestyle=':', label='Stagnation Enthalpy, $h_0$')
        h_axis.set_xticks(stages)
        h_axis.set_xlabel('Stage')
        h_axis.set_ylabel('Enthalpy, kJ/kg')
        h_axis.set_xlim(0, 5)
        h_axis.legend()

        rho_fig, rho_axis = plt.subplots()
        rho_stages = np.array([self.rho_inf, self.rho_1, self.rho_2, self.rho_3,
                               self.rho_4, self.rho_5])
        rho0_stages = np.array([self.rho_0inf, self.rho_01, self.rho_02, self.rho_03,
                                self.rho_04, self.rho_05])
        rho_axis.plot(stages, rho_stages, color='k', label='Static Density, $\rho$')
        rho_axis.plot(stages, rho0_stages, color='k', linestyle=':', label='Stagnation Density, $\rho_0$')
        rho_axis.set_xticks(stages)
        rho_axis.set_xlabel('Stage')
        rho_axis.set_ylabel(r'Density, $\mathrm{kg/m^3}$')
        rho_axis.set_xlim(0, 5)
        rho_axis.legend()

        v_fig, v_axis = plt.subplots()
        v_axis.plot(stages, 1./rho_stages, color='k', label='Static Specific Volume, $v$')
        v_axis.plot(stages, 1./rho0_stages, color='k', linestyle=':', label='Stagnation Specific Volume, $v_0$')
        v_axis.set_xticks(stages)
        v_axis.set_xlabel('Stage')
        v_axis.set_ylabel(r'Specific Volume, $\mathrm{m^3/kg}$')
        v_axis.set_xlim(0, 5)
        v_axis.legend()

        pv_fig, pv_axis = plt.subplots()
        pv_axis.scatter(p_stages/1000., 1./rho_stages, color='k', facecolor='none')
        pv_axis.plot(p_stages/1000., 1./rho_stages, color='k', linestyle=':')
        pv_axis.set_xlabel('Pressure, kPa')
        pv_axis.set_ylabel(r'Specific Volume, $\mathrm{m^3/kg}$')

        for i, txt in enumerate(stages):
            pv_axis.annotate(txt, (p_stages[i]/1000. +10., 1./rho_stages[i] + 0.1))

        Ts_fig, Ts_axis = plt.subplots()
        Ts_axis.scatter(T_stages, s_stages/1000., color='k', facecolor='none')
        Ts_axis.plot(T_stages, s_stages/1000., color='k', linestyle=':')
        Ts_axis.set_xlabel('Temperature, K')
        Ts_axis.set_ylabel('Specific Entropy, kJ/(kg-K)')

        for i, txt in enumerate(stages):
            if i == 0:
                Ts_axis.annotate(r'$\infty$', (T_stages[i] - 15., s_stages[i]/1000. + 0.03))
            else:
                Ts_axis.annotate(txt, (T_stages[i] +15., s_stages[i]/1000. + 0.03))
        plots = [T_fig, P_fig, M_fig, s_fig, h_fig, rho_fig, v_fig, pv_fig, Ts_fig]
        pdf = matplotlib.backends.backend_pdf.PdfPages("Ideal_Turbojet_Christian_Bolander.pdf")
        for fig in plots: ## will open an empty extra figure :(
            pdf.savefig( fig )
        pdf.close()


if __name__ == "__main__":
    plt.close('all')
    jet = turbojet()
    pdf = jet.output()
    jet.plot_states(pdf)
