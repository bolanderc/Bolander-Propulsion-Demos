#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 21:01:50 2021

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
import matplotlib.backends.backend_pdf

class Turbojet:
    def __init__(self):
        self.M_inf = 0.8
        self.T_inf = 216.65  # K
        self.p_inf = 22.7e3  # Pa
        self.T_04 = 1944.  # K
        self.pi_c = 20.
        self.A2_A4star = 10.
        self.A2_A1throat = 1.2
        self.M_4 = 1.
        self.M_8 = 1.
        self.A_1throat = 20./(100**2)  # m^2
        self.f = 50.
        self.gamma = 1.4
        self.pi_b = 1.0
        self.pi_d = 1.0
        self.Aexit_A8 = 1.0
        # self.M_inf = 0.85
        # self.T_inf = 216.65
        # self.p_inf = 22.63e3
        # self.T_04 = 2023.15
        # self.pi_c = 15.
        # self.A2_A4star = 8
        # self.A2_A1throat = 1.5
        # self.A_1throat = 10./(100**2)
        # self.Aexit_A8 = 2.
        # self.pi_b = 1.0
        # self.M_4 = 1.
        # self.M_8 = 1.
        # self.pi_d = 1.0
        # self.f = 40.
        # self.gamma = 1.4
        self.R_g = 287.056  # J/(kg-K)
        self.p_sl = 101.325e3  # Pa
        self.T_sl = 288.15  # K
        self.g_0 = 9.8067  # m/s^2
        self.c_p = self.gamma*self.R_g/(self.gamma - 1)  # J/kg-K
        self.M1t_estimate = 0.9*self.M_inf

    def _tau_lambda(self):
        tau_lambda = self.T_04/self.T_inf
        return tau_lambda

    def _tau_compressor(self):
        exp = (self.gamma - 1.)/self.gamma
        tau_comp = self.pi_c**exp
        return tau_comp

    def _tau_reference(self):
        self.T_0inf = comp_codes.isenflowprops("T", self.M_inf,
                                               gamma=self.gamma,
                                               flow=self.T_inf)
        self.P_0inf = comp_codes.isenflowprops("P", self.M_inf,
                                               gamma=self.gamma,
                                               flow=self.p_inf)
        tau_ref = self.T_0inf/self.T_inf
        return tau_ref

    def _tau_turbine(self):
        C1 = self.f/(self.f + 1.)
        C2 = self.tau_c - 1.
        C3 = self.tau_r/self.tau_lambda
        tau_turb = 1 - C1*C2*C3
        return tau_turb

    def _FM(self, M):
        num = M
        D1 = (1. + (self.gamma - 1.)/2.*M*M)
        D2 = (2./(self.gamma + 1.))
        Dexp = 0.5*(self.gamma + 1.)/(self.gamma - 1.)
        return num/((D1*D2)**Dexp)

    def _FM2(self, mdot_c):
        C1 = self.A_2*self.p_sl
        C2 = np.sqrt(self.gamma/self.R_g/self.T_sl)
        C3 = np.sqrt((2./(self.gamma + 1.))**((self.gamma + 1.)/(self.gamma - 1.)))
        return mdot_c/(C1*C2*C3)

    def _compressor_demand_match(self):
        C1 = (self.f/(self.f + 1.))
        C2 = self.pi_c*self.pi_b/np.sqrt(self.tau_lambda/self.tau_r)
        C3 = 1./self.A2_A4star
        self.A2_A2star = 1./(C1*C2*C3)
        M_2 = optimize.minimize(comp_codes.area_mach, self.M1t_estimate/2., args=(self.A2_A2star, self.gamma), method='Nelder-Mead').x[0]
        return M_2

    def _nozzle_turbine_match(self):
        exp = 0.5*((self.gamma + 1.)/(self.gamma - 1.))
        self.A4star_A8 = self.tau_t**exp

    def _inlet_flow_match(self):
        FMinf = self._FM(self.M_inf)
        FM2 = self._FM(self.M_2)
        self.Ainf_A1throat = self.pi_d*FM2/FMinf*self.A2_A1throat

    def _exit_ratios(self):
        num = 1. + 0.5*(self.gamma - 1.)*self.M_inf**2
        denom = 1. + 0.5*(self.gamma - 1.)*self.M_exit**2
        denom_star = 1 + 0.5*(self.gamma - 1.)
        exp = self.gamma/(self.gamma - 1.)
        self.Texit_Tinf = self.tau_lambda*self.tau_t*num/self.tau_r/denom
        self.pexit_pinf = self.pi_c*(self.tau_t*num/denom)**exp
        self.p8star_pinf = self.pi_c*(self.tau_t*num/denom_star)**exp

    def _thrust_Isp(self):
        self.T_mom = self.gamma*self.M_inf**2*((self.f + 1.)/self.f*self.Vexit_Vinf - 1.)
        self.T_press = (self.pexit_pinf - 1.)*self.Aexit_Ainf
        self.T_total = self.T_mom + self.T_press
        self.I = self.T_total*self.f/(self.gamma*self.M_inf)
        self.F_T_total = self.T_total*self.p_inf*self.A_inf
        self.Isp_total = self.F_T_total/(self.g_0*self.mdot_fuel)
        self.TSFC = 1./(self.g_0*self.Isp_total)*2.204*4.4495*3600.


    def static_nozzle_analysis(self, optimal_value=1., optimal_nozzle=False,
                               pi_choke=False, pi_c=20.):
        if optimal_nozzle:
            self.Aexit_A8 = optimal_value
        if pi_choke:
            self.pi_c = optimal_value
        self.tau_lambda = self._tau_lambda()
        self.tau_c = self._tau_compressor()
        self.tau_r = self._tau_reference()
        self.tau_t = self._tau_turbine()
        self.M_2 = self._compressor_demand_match()
        self._nozzle_turbine_match()
        self.A1throat_A2star = self.A2_A2star/self.A2_A1throat
        self.M_1throat = optimize.minimize(comp_codes.area_mach, self.M1t_estimate, args=(self.A1throat_A2star,
                                                                            self.gamma), method='Nelder-Mead').x[0]
        self._inlet_flow_match()
        self.M_exit = optimize.minimize(comp_codes.area_mach, 2.5,
                                        args=(self.Aexit_A8, self.gamma)).x[0]
        self._exit_ratios()
        self.Vexit_Vinf = self.M_exit/self.M_inf*np.sqrt(self.Texit_Tinf)
        self.A_inf = self.Ainf_A1throat*self.A_1throat
        self.Aexit_Ainf = self.Aexit_A8/self.A4star_A8/self.A2_A4star*self.A2_A1throat/self.Ainf_A1throat
        self.mdot_inf = np.sqrt(self.gamma/(self.R_g*self.T_inf))*self.p_inf*self.A_inf*self.M_inf
        self.mdot_fuel = self.mdot_inf/self.f
        self.mdot_exit = self.mdot_inf*(self.f + 1.)/self.f
        self.mdot_corrected = self.mdot_inf*np.sqrt(self.T_0inf/self.T_sl)/(self.P_0inf/self.p_sl)
        self._thrust_Isp()
        self.V_inf = self.M_inf*np.sqrt(self.gamma*self.R_g*self.T_inf)
        self.h_0inf = self.c_p*self.T_inf + 0.5*self.V_inf**2
        self.V_exit = self.Vexit_Vinf*self.V_inf
        self.T_exit = comp_codes.isenflowprops("T", self.M_exit,
                                               gamma=self.gamma,
                                               stagnation=self.T_04)
        self.h_0exit = self.c_p*self.T_exit + 0.5*self.V_exit**2
        self.h_fuel = self.c_p*self.T_inf*(self.tau_lambda + self.f*(self.tau_lambda - self.tau_r*self.tau_c))
        if optimal_nozzle:
            return abs(self.pexit_pinf - 1.)
        if pi_choke:
            return abs(self.M_1throat - 1.)


    def expandable_nozzle_analysis(self):
        Ae_A8star_opt = optimize.minimize(self.static_nozzle_analysis, 2.5, args=(True)).x[0]
        pexit_pinf_opt = self.static_nozzle_analysis(optimal_value=Ae_A8star_opt,
                                                      optimal_nozzle=True)
        assert(abs(pexit_pinf_opt)<1e-6)
        self.A_2 = self.A2_A1throat*self.A_1throat
        print("\n----------")
        print("| Part 2 |")
        print("----------")
        print(f"a) Optimal expansion ratio for nozzle : {case.Aexit_A8:1.4f}")
        print(f"b) Velocity ratio across engine : {case.Vexit_Vinf:1.4f}")
        print(f"c) Total thrust : {case.F_T_total:1.4f} N")
        print(f"   Specific Impulse : {case.Isp_total:1.4f} s")
        print(f"   Thrust-specific fuel consumption : {case.TSFC:1.4f} lbm/(lbf-hr)")

        self.pi_c_choke = optimize.minimize(self.static_nozzle_analysis, 23.975, args=(False, True), method='Nelder-Mead').x[0]
        M1_throat_choke = self.static_nozzle_analysis(optimal_value=self.pi_c_choke, pi_choke=True)
        assert (abs(M1_throat_choke)<1e-4)
        pic_range = np.linspace(1, self.pi_c_choke, 200)
        fM2_range = np.zeros_like(pic_range)
        mdotc_range = np.zeros_like(pic_range)
        M1t_range = np.zeros_like(pic_range)
        Ainf_range = np.zeros_like(pic_range)
        for i in range(len(pic_range)):
            self.pi_c = pic_range[i]
            if i == 0:
                self.M1t_estimate = 0.05
            else:
                self.M1t_estimate = self.M_1throat
            self.static_nozzle_analysis()
            mdotc_range[i] = self.mdot_corrected
            fM2_range[i] = self._FM2(mdotc_range[i])
            M1t_range[i] = self.M_1throat
            Ainf_range[i] = self.A_inf
        pm_fig, pm_ax = plt.subplots()
        pm_ax.plot(mdotc_range*1000, pic_range, color='k')
        pm_ax.set_ylim(1, self.pi_c_choke)
        pm_ax.set_yticks(np.arange(1., self.pi_c_choke + 1., 2.))
        pm_ax.set_xlim(0, 500.)
        pm_ax.set_xticks(np.arange(0, 525., 50.))
        pm_ax.tick_params(axis='x', labelsize=14)
        pm_ax.tick_params(axis='y', labelsize=14)
        pm_ax.set_ylabel(r"$\pi_c$", fontsize=14)
        pm_ax.set_xlabel(r"$\dot{m}_c$, g/s", fontsize=14)
        pm_ax.grid()
        pm_fig.tight_layout()
        pfm_fig, pfm_ax = plt.subplots()
        pfm_ax.plot(fM2_range, pic_range, color='k')
        pfm_ax.set_ylim(1, self.pi_c_choke)
        pfm_ax.set_yticks(np.arange(1., self.pi_c_choke + 1., 2.))
        pfm_ax.set_xlim(0, 0.85)
        pfm_ax.set_xticks(np.arange(0, 0.9, 0.1))
        pfm_ax.tick_params(axis='x', labelsize=14)
        pfm_ax.tick_params(axis='y', labelsize=14)
        pfm_ax.set_ylabel(r"$\pi_c$", fontsize=14)
        pfm_ax.set_xlabel(r"$f(M_2)$", fontsize=14)
        pfm_ax.grid()
        pfm_fig.tight_layout()
        pM1t_fig, pM1t_ax = plt.subplots()
        pM1t_ax.plot(M1t_range, pic_range, color='k')
        pM1t_ax.set_ylim(1, 25.)
        pM1t_ax.set_yticks(np.arange(1., 26., 2.))
        pM1t_ax.set_xlim(0, 1.0)
        pM1t_ax.set_xticks(np.arange(0, 1.1, 0.1))
        pM1t_ax.tick_params(axis='x', labelsize=14)
        pM1t_ax.tick_params(axis='y', labelsize=14)
        pM1t_ax.set_ylabel(r"$\pi_c$", fontsize=14)
        pM1t_ax.set_xlabel(r"$M_{1,\mathrm{throat}}$", fontsize=14)
        pM1t_ax.grid()
        pM1t_fig.tight_layout()

        Ainfm_fig, Ainfm_ax = plt.subplots()
        Ainfm_ax.plot(mdotc_range*1000, Ainf_range*100.**2, color='k')
        Ainfm_ax.set_xlim(0, 500.)
        Ainfm_ax.set_xticks(np.arange(0, 525., 50.))
        Ainfm_ax.tick_params(axis='x', labelsize=14)
        Ainfm_ax.tick_params(axis='y', labelsize=14)
        Ainfm_ax.set_ylabel(r"$A_\infty, \mathrm{cm^2}$", fontsize=14)
        Ainfm_ax.set_xlabel(r"$\dot{m}_c$, g/s", fontsize=14)
        Ainfm_ax.grid()
        Ainfm_ax.axhline(self.A_1throat*100.**2, color='k', linestyle=':')
        Ainfm_ax.annotate(r"$A_{1,\mathrm{throat}}$", (250., 20.5), fontsize=14)
        Ainfm_fig.tight_layout()

        Ainffm_fig, Ainffm_ax = plt.subplots()
        Ainffm_ax.plot(fM2_range, Ainf_range*100.**2, color='k')
        Ainffm_ax.set_xlim(0, 0.85)
        Ainffm_ax.set_xticks(np.arange(0, 0.9, 0.1))
        Ainffm_ax.tick_params(axis='x', labelsize=14)
        Ainffm_ax.tick_params(axis='y', labelsize=14)
        Ainffm_ax.set_ylabel(r"$A_\infty, \mathrm{cm^2}$", fontsize=14)
        Ainffm_ax.set_xlabel(r"$f(M_2)$", fontsize=14)
        Ainffm_ax.grid()
        Ainffm_ax.axhline(self.A_1throat*100.**2, color='k', linestyle=':')
        Ainffm_ax.annotate(r"$A_{1,\mathrm{throat}}$", (0.425, 20.5), fontsize=14)
        Ainffm_fig.tight_layout()

        AinfM1t_fig, AinfM1t_ax = plt.subplots()
        AinfM1t_ax.plot(M1t_range, Ainf_range*100.**2, color='k')
        AinfM1t_ax.set_xlim(0, 1.0)
        AinfM1t_ax.set_xticks(np.arange(0, 1.1, 0.1))
        AinfM1t_ax.tick_params(axis='x', labelsize=14)
        AinfM1t_ax.tick_params(axis='y', labelsize=14)
        AinfM1t_ax.set_ylabel(r"$A_\infty, \mathrm{cm^2}$", fontsize=14)
        AinfM1t_ax.set_xlabel(r"$M_{1,\mathrm{throat}}$", fontsize=14)
        AinfM1t_ax.grid()
        AinfM1t_ax.axhline(self.A_1throat*100.**2, color='k', linestyle=':')
        AinfM1t_ax.annotate(r"$A_{1,\mathrm{throat}}$", (0.5, 20.5), fontsize=14)
        AinfM1t_fig.tight_layout()

        plots = [pm_fig, pfm_fig, pM1t_fig, Ainfm_fig, Ainffm_fig, AinfM1t_fig]
        pdf = matplotlib.backends.backend_pdf.PdfPages("Turbojet_Nozzle_Design_Figures_Christian_Bolander.pdf")
        for fig in plots: ## will open an empty extra figure :(
            pdf.savefig( fig )
        pdf.close()





if __name__ == "__main__":
    plt.close('all')
    case = Turbojet()
    case.static_nozzle_analysis()
    print("----------------------------------")
    print("| Homework 5.4 Analysis Part (1) |")
    print("----------------------------------\n")
    print(f"a) True compressor massflow : {case.mdot_inf*1000.:1.4f} g/s")
    print(f"   Corrected compressor massflow : {case.mdot_corrected*1000:1.4f} g/s")
    print(f"b) Normalized exit pressure thrust : {case.T_press:1.4f}")
    print(f"   Normalized momentum thrust : {case.T_mom:1.4f}")
    print(f"   Normalized total thrust : {case.T_total:1.4f}")
    print(f"c) Velocity ratio across the engine : {case.Vexit_Vinf:1.4f}")
    print(f"d) Mach number at diffuser throat : {case.M_1throat:1.4f}")
    print(f"e) Inlet capture area : {case.A_inf*100**2:1.4f} cm^2")
    print(f"f) Total thrust : {case.F_T_total:1.4f} N")
    print(f"   Specific Impulse : {case.Isp_total:1.4f} s")
    print(f"   Thrust-specific fuel consumption : {case.TSFC:1.4f} lbm/(lbf-hr)")
    print(f"g) Enthalpy of the fuel : {case.h_fuel/1e6:1.4f} MJ/kg")
    case.expandable_nozzle_analysis()
    print("d) Assuming the same combustor temperature and inlet throat area:")
    print(f"   Choking compressor compression ratio : {case.pi_c_choke:1.4f}")

