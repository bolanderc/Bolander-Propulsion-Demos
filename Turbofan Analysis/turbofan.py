#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:43:00 2021

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
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

class Turbofan:
    def __init__(self):
        self.M_inf = 0.7
        self.gamma = 1.4
        self.p_inf = 47218.  # Pa
        self.a_inf = 316.4517  # m/s
        self.V_inf = self.M_inf*self.a_inf  # m/s
        self.T_inf = 249.1868  # K
        self.R_g = 287.056  # J/(kg-K)
        self.p_sl = 101.325e3  # Pa
        self.T_sl = 288.15  # K
        self.g_0 = 9.8067  # m/s^2
        self.c_p = 1004.96  # J/kg-K
        self.T_04 = 1700.  # K
        self.h_f = 42.68e6  # J/kg
        self.eta_b = 0.35
        self.pi_c = 6.
        self.pi_fan = 2.
        self.tau_f = self.eta_b*self.h_f/(self.c_p*self.T_inf)
        self.beta = 2.
        self.B = self.beta/(self.beta + 1.)

    def _tau_lambda(self):
        tau_lambda = self.T_04/self.T_inf
        return tau_lambda

    def _tau_compressor(self):
        exp = (self.gamma - 1.)/self.gamma
        tau_comp = self.pi_c**exp
        tau_fan = self.pi_fan**exp
        return tau_comp, tau_fan

    def _tau_reference(self):
        self.T_0inf = comp_codes.isenflowprops("T", self.M_inf,
                                               gamma=self.gamma,
                                               flow=self.T_inf)
        self.P_0inf = comp_codes.isenflowprops("P", self.M_inf,
                                               gamma=self.gamma,
                                               flow=self.p_inf)
        tau_ref = self.T_0inf/self.T_inf
        return tau_ref

    def _air_to_fuel(self):
        f = (1. + self.beta)*(self.tau_f - self.tau_lambda)/(self.tau_lambda - self.tau_c*self.tau_r)
        return f

    def _tau_turbine(self):
        C1 = self.tau_r/self.tau_lambda
        num = (self.tau_c - 1.) + self.beta*(self.tau_fan - 1.)
        denom = (1. + (1. + self.beta)/self.f)
        tau_turb = 1. - C1*num/denom
        return tau_turb

    def _velocity_ratios(self):
        num_fan = self.tau_r*self.tau_fan - 1.
        denom_fan = self.tau_r - 1.
        vratio_fan = np.sqrt(num_fan/denom_fan)
        num_core = (self.tau_r*self.tau_c*self.tau_t - 1.)*self.tau_lambda
        denom_core = (self.tau_r - 1.)*(self.tau_c*self.tau_r)
        vratio_core = np.sqrt(num_core/denom_core)
        return vratio_fan, vratio_core

    def _thrust(self):
        C1_fan = self.gamma*self.M_inf**2*self.beta/(self.beta + 1.)
        C2_fan = np.sqrt((self.tau_r*self.tau_fan - 1.)/(self.tau_r - 1.)) - 1.
        self.T_mom_fan = C1_fan*C2_fan
        C1_core = self.gamma*self.M_inf**2
        C2_core = (1. + (1./self.f)*(1. + self.beta))/(1. + self.beta)
        C3_core = (self.tau_r*self.tau_c*self.tau_t - 1.)*self.tau_lambda
        C4_core = (self.tau_r - 1.)*(self.tau_c*self.tau_r)
        self.T_mom_core = C1_core*C2_core*(np.sqrt(C3_core/C4_core) - 1.)
        self.T_total = self.T_mom_fan + self.T_mom_core
        self.I_total = self.T_total*self.f/(self.gamma*self.M_inf)
        self.Isp_total = self.I_total*self.a_inf/self.g_0
        self.TSFC = 1./(self.g_0*self.Isp_total)*2.204*4.4495*3600.

    def _optimal_bypass(self):
        C1 = 1./(self.tau_fan - 1.)
        C2 = self.tau_lambda/(self.tau_c*self.tau_r) - 1.
        C3 = self.tau_c - 1.
        C4 = self.tau_lambda*(self.tau_r - 1.)/(self.tau_r**2*self.tau_c)
        C5 = 1./4.*(self.tau_r - 1.)/self.tau_r
        C6 = np.sqrt((self.tau_r*self.tau_fan - 1.)/(self.tau_r - 1.)) + 1.
        beta_optimal = C1*(C2*C3 + C4 - C5*C6)
        return beta_optimal

    def _propulsive_efficiency(self):
        num1 = (1./(1. + self.beta))*(self.vratio_core - 1.)
        num2 = (self.beta/(self.beta + 1.))*(self.vratio_fan - 1.)
        denom1 = (1 + self.f)/self.f*self.vratio_core**2
        denom2 = self.beta*self.vratio_fan**2
        denom3 = 1. + self.beta
        num = 2.*(num1 + num2)
        denom = denom1 + denom2 - denom3
        eta_propulsive = num/denom
        return eta_propulsive

    def _thermal_efficiency(self):
        eta_thermal = 1. - 1./(self.tau_c*self.tau_r)
        return eta_thermal

    def analysis(self):
        self.tau_lambda = self._tau_lambda()
        self.tau_c, self.tau_fan = self._tau_compressor()
        self.tau_r = self._tau_reference()
        self.f = self._air_to_fuel()
        self.tau_t = self._tau_turbine()
        self.vratio_fan, self.vratio_core = self._velocity_ratios()
        self.Ve_fan = self.vratio_fan*self.V_inf
        self.Ve_core = self.vratio_core*self.V_inf
        self.Ve_increment = (self.Ve_fan - self.V_inf)/(self.Ve_core - self.V_inf)
        self._thrust()
        self.eta_propulsive = self._propulsive_efficiency()
        self.eta_thermal = self._thermal_efficiency()
        self.eta_total = self.eta_propulsive*self.eta_thermal


if __name__ == "__main__":
    plt.close('all')
    case = Turbofan()
    case.analysis()
    beta_optimal_analytic = case._optimal_bypass()
    eta_p_original = case.eta_propulsive
    eta_t_original = case.eta_thermal
    eta_total_original = case.eta_total
    print("-------------------------")
    print("| Homework 6.2 Analysis |")
    print("-------------------------\n")
    print("Non-dimensional Parameters:")
    print(f"tau_lambda : {case.tau_lambda:1.4f}")
    print(f"tau_c_core : {case.tau_c:1.4f}")
    print(f"tau_c_fan : {case.tau_fan:1.4f}")
    print(f"tau_r : {case.tau_r:1.4f}")
    print(f"Air-to-Fuel Ratio : {case.f:1.4f}")
    print(f"tau_t : {case.tau_t:1.4f}")
    print(f"Ve_fan/V_inf : {case.vratio_fan:1.4f}")
    print(f"Ve_core/V_inf : {case.vratio_core:1.4f}\n")
    print(f"i) Normalized Thrust : {case.T_total:1.4f}")
    print(f"ii) Percent Thrust Delivered by Core Flow : {case.T_mom_core/case.T_total*100:1.4f} %")
    print(f"iii) Percent Thrust Delivered by Bypass Flow : {case.T_mom_fan/case.T_total*100:1.4f} %")
    print(f"iv) Ratio of Bypass Thrust to Core Thrust : {case.T_mom_fan/case.T_mom_core:1.4f}")
    print(f"v) Normalized Specific Impulse : {case.I_total:1.4f}")
    print(f"vi) Thrust-Specific-Fuel-Consumption (TSFC) : {case.TSFC:1.4f} lbm/lbf-hr")
    print(f"vii) Bypass Ratio for Optimal Specific Impulse : {beta_optimal_analytic:1.4f}")
    case.beta = beta_optimal_analytic
    case.analysis()
    print(f"viii) Optimal TSFC : {case.TSFC:1.4f} lbm/lbf-hr")
    print(f"ix) Efficiencies From Original Analysis:")
    print(f"    Thermal : {eta_t_original*100.:1.4f} %")
    print(f"    Propulsive : {eta_p_original*100.:1.4f} %")
    print(f"    Total : {eta_total_original*100.:1.4f} %")
    print(f"x) Efficiencies From Optimal Bypass Analysis:")
    print(f"   Thermal : {case.eta_thermal*100.:1.4f} %")
    print(f"   Propulsive : {case.eta_propulsive*100.:1.4f} %")
    print(f"   Total : {case.eta_total*100.:1.4f} %")

    N = 1000
    bypass_range = np.linspace(0.1, 25, N)
    T_fan = np.zeros(N)
    T_core = np.zeros(N)
    T_total = np.zeros(N)
    I_range = np.zeros(N)
    Ve_ratio_fan = np.zeros(N)
    Ve_ratio_core = np.zeros(N)
    Ve_increment = np.zeros(N)
    f_range = np.zeros(N)
    for i in range(N):
        case.beta = bypass_range[i]
        case.analysis()
        T_fan[i] = case.T_mom_fan
        T_core[i] = case.T_mom_core
        T_total[i] = case.T_total
        I_range[i] = case.I_total
        Ve_ratio_fan[i] = case.vratio_fan
        Ve_ratio_core[i] = case.vratio_core
        Ve_increment[i] = case.Ve_increment
        f_range[i] = case.f

    fig1 = plt.figure()
    plt.plot(bypass_range, I_range, color='k')
    plt.xlabel(r'Bypass Ratio, $\beta$')
    plt.ylabel(r'Normalized Specific Impulse, $\mathbb{I}$')
    plt.xlim(0, 12)
    plt.ylim(30, 70)
    plt.grid()

    fig2 = plt.figure()
    plt.plot(bypass_range, T_core, color='k', linestyle='-.', label='Core Thrust')
    plt.plot(bypass_range, T_fan, color='k', linestyle=':', label='Bypass Thrust')
    plt.plot(bypass_range, T_total, color='k', linestyle='-', label='Total Thrust')
    plt.xlabel(r'Bypass Ratio, $\beta$')
    plt.ylabel(r'Normalized Thrust Components, $\mathbb{T}$')
    plt.xlim(0, 12)
    plt.ylim(0, )
    plt.legend()
    plt.grid()

    fig3 = plt.figure()
    plt.plot(bypass_range, Ve_ratio_core, color='k', linestyle='-.', label='Core')
    plt.plot(bypass_range, Ve_ratio_fan, color='k', linestyle=':', label='Bypass')
    plt.plot(bypass_range, Ve_increment, color='k', linestyle='-', label=r'$\frac{V_{e,\mathrm{fan}} - V_\infty}{V_{e,\mathrm{core}} - V_\infty}$')
    plt.axhline(2., color='r', linestyle='-', label='Optimum')
    plt.xlabel(r'Bypass Ratio, $\beta$')
    plt.ylabel(r'Velocity Ratio')
    plt.xlim(0, 12)
    plt.ylim(-10, 10)
    plt.legend()
    plt.grid()

    fig4 = plt.figure()
    plt.plot(bypass_range, f_range, color='k')
    plt.xlabel(r'Bypass Ratio, $\beta$')
    plt.ylabel(r'Air-to-Fuel Ratio, $f$')
    plt.xlim(0, 12)
    plt.ylim(0, 150)
    plt.grid()

    plots = [fig1, fig2, fig3, fig4]
    pdf = matplotlib.backends.backend_pdf.PdfPages("Turbofan_Design_Figures_Christian_Bolander.pdf")
    for fig in plots: ## will open an empty extra figure :(
        pdf.savefig( fig )
    pdf.close()

    beta_optimal_numerical = bypass_range[np.nanargmax(I_range)]
    print(f"xi) Numerically Evaluated Optimal Bypass Ratio : {beta_optimal_numerical:1.4f}")


