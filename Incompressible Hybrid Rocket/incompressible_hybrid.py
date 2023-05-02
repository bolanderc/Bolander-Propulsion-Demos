#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 08:51:37 2021

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import sys
sys.path.insert(1, '/home/christian/Python Projects/School Work/Fall 2019/MAE 5420')
import KobayashiMaru
comp_codes = KobayashiMaru.KobayashiMaru()
import scipy.optimize as optimize

class incompressible_hybrid:
    def __init__(self, p_inj):
        self.T_desired = 8.0e3  # Desired thrust, kN
        self.exp_ratio = 16.4
        self.D_e = 19.17e-2  # m
        self.A_e = np.pi*self.D_e*self.D_e/4.
        self.theta_e = np.deg2rad(10.)  # rad
        self.D_c_t0 = 11.0e-2  # m
        self.L_P = 100.0e-2  # m
        self.p_amb = 60.0e3  # Pa
        self.p_sl = 101.325e3  # Pa
        self.D_t = 4.734e-2  # m
        self.A_star = np.pi*self.D_t*self.D_t/4.
        self.p_inj = p_inj  # Pa
        self.rho_ox = 892.  # kg/m^3
        self.g_0 = 9.8067  # m/s^2
        self.Pr_bl = 0.5
        self.eps_combustor = 0.99
        self.rho_fuel = 930.  # kg/m^3
        self.h_v_fuel_surface = 1.8e6  # J/kg
        self.T_fuel_surface = 300.  # K


        self.N_P_inj = 50
        self.D_P_inj = 2.0e-3  # m
        self.C_P_d = 0.81
        self.A_inj = self.N_P_inj*np.pi*self.D_P_inj*self.D_P_inj/4.  # m^2

        self.O_F_ratio = np.array([2., 3., 3.5, 4., 5., 5.5, 6., 7., 8., 9.])
        self.T_0_of = np.array([2019.4, 2329.5, 2643.2, 2895.8, 3225.5,
                                3315., 3366.3, 3399.4, 3388.3, 3359.4])  # K
        self.gamma_of = np.array([1.251, 1.256, 1.28, 1.256, 1.208, 1.185,
                                  1.167, 1.148, 1.142, 1.141])
        self.M_W_of = np.array([20.494, 20.823, 21.653, 22.779, 24.613,
                                25.332, 25.925, 26.807, 27.421, 27.871])  # kg/kg-mol
        self.R_u = 8314.  # kJ/(kg mol K)
        self.R_g_of = self.R_u/self.M_W_of  # kJ/(kg K)
        self.C_p_of = self.gamma_of*self.R_g_of/(self.gamma_of - 1.)  # J/(kg K)
        self.eta_t = 0.99**2

        self.f_T0_of = interp1d(self.O_F_ratio, self.T_0_of, kind='cubic',fill_value='extrapolate')
        self.f_gamma_of = interp1d(self.O_F_ratio, self.gamma_of, kind='cubic',fill_value='extrapolate')
        self.f_MW_of = interp1d(self.O_F_ratio, self.M_W_of, kind='cubic',fill_value='extrapolate')
        self.f_Rg_of = interp1d(self.O_F_ratio, self.R_g_of, kind='cubic',fill_value='extrapolate')
        self.f_Cp_of = interp1d(self.O_F_ratio, self.C_p_of, kind='cubic',fill_value='extrapolate')

    def _southerland(self, O_F):
        T_0 = self.f_T0_of(O_F)[()]*self.eta_t
        T_s = 300.  # K
        C_s = 240.  # K
        mu_s = 1.4889e-5  # Pa-sec
        E1 = 3./2.
        C1 = T_0/T_s
        C2 = (T_s + C_s)/(T_0 + C_s)
        mu_T0 = mu_s*C1**E1*C2
        return mu_T0

    def _A_burn(self, R_c):
        return 2.*np.pi*R_c*self.L_P

    def _A_chamber(self, R_c):
        return np.pi*R_c*R_c

    def _V_chamber(self, R_c):
        return np.pi*R_c*R_c*self.L_P

    def _mdot_ox(self, P_0):
        C1 = self.C_P_d*self.A_inj
        C2 = np.sqrt(2.*self.rho_ox*(self.p_inj - P_0))
        return C1*C2

    def _mdot_fuel(self, R_c, rdot_fuel):
        A_burn = self._A_burn(R_c)
        return self.rho_fuel*A_burn*rdot_fuel

    def _G_ox(self, P_0, R_c, rdot_fuel):
        mdot_ox = self._mdot_ox(P_0)
        mdot_fuel = self._mdot_fuel(R_c, rdot_fuel)
        G_ox = mdot_ox/mdot_fuel
        return G_ox

    def _dh_flame(self, O_F):
        T_flame = self.f_T0_of(O_F)[()]
        C_p_fuel = self.f_Cp_of(O_F)[()]
        dh_flame = C_p_fuel*(T_flame - self.T_fuel_surface)
        return dh_flame

    def _D_c_t(self, t):
        rdot_int = np.trapz(self.rdot_t, dx=self.dt)
        D_c_t = self.D_c_t0 + 2.*rdot_int
        return D_c_t

    def _port_regression_rate(self, P_0, R_c, O_F, total):
        mu_fuel = self._southerland(O_F)
        C1 = 0.047/(self.Pr_bl**(2./3.)*self.rho_fuel)
        dh_flame = self._dh_flame(O_F)
        C2 = dh_flame/self.h_v_fuel_surface
        E1 = 0.23
        mdot_ox = self._mdot_ox(P_0)
        A_c = self._A_chamber(R_c)
        E2 = 4./5.
        C4 = mu_fuel/self.L_P
        E3 = 1./5.
        if not total:
            C3 = mdot_ox/A_c
            return C1*(C2**E1)*(C3**E2)*(C4**E3)
        else:
            C3_1 = (mdot_ox/A_c)**E3
            C3_2 = (5./9.)*(0.047/(self.Pr_bl)**E1)
            C3_3 = C2**E1
            C3_4 = C4**E3
            C3_5 = self.L_P/(2.*R_c)
            C3 = (C3_1 + C3_2*C3_3*C3_4*C3_5)
            E4 = 4.
            return C1*(C2**E1)*(C3**E4)*(C4**E3)

    def _M_O_F(self, OF_m1, t, mdot_ox):
        mu_fuel = self._southerland(OF_m1)
        C1 = 5.58244*self.Pr_bl**(2./3.)
        dh_flame = self._dh_flame(OF_m1)
        C2 = (self.h_v_fuel_surface/dh_flame)
        E1 = 0.23
        C3 = (mdot_ox/(mu_fuel*self.L_P))
        E2 = 1./5.
        D_c_t = self._D_c_t(t)
        C4 = D_c_t/self.L_P
        E3 = 3./5.
        return C1*C2**E1*C3**E2*C4**E3

    def _mdot_exit(self, P_0, O_F):
        gamma = self.f_gamma_of(O_F)[()]
        T_0 = self.f_T0_of(O_F)[()]
        R_g = self.f_Rg_of(O_F)[()]
        A_star = self.A_star
        C1 = P_0*A_star
        C2 = gamma/(R_g*T_0)
        C3 = (2./(gamma + 1.))
        E1 = (gamma + 1.)/(gamma - 1.)
        return C1*np.sqrt(C2*C3**E1)

    def _c_star(self, O_F, T_0):
        gamma = self.f_gamma_of(O_F)[()]
        R_g = self.f_Rg_of(O_F)[()]
        C1 = np.sqrt(gamma*R_g*T_0)
        C2 = (2./(gamma + 1.))
        E1 = (gamma + 1.)/(gamma - 1.)
        return C1/(gamma*np.sqrt(C2**E1))

    def _slthrust_eq(self, F_vac):
        return F_vac - self.A_e*self.p_sl

    def _vacthrust_eq(self, P_0, O_F):
        mdot_e = self._mdot_exit(P_0, O_F)
        gamma = self.f_gamma_of(O_F)[()]
        T_0_actual = self.f_T0_of(O_F)[()]
        R_g = self.f_Rg_of(O_F)[()]
        M_e = comp_codes.area_mach(1.4, A_Astar=self.exp_ratio, gamma=gamma)
        p_e = comp_codes.isenflowprops("P", M_e, gamma=gamma, stagnation=P_0)
        T_e = comp_codes.isenflowprops("T", M_e, gamma=gamma, stagnation=T_0_actual)
        lam_t = 0.5*(1. + np.cos(self.theta_e))
        C1 = lam_t*mdot_e*M_e
        C2 = np.sqrt(gamma*R_g*T_e)
        C3 = self.A_e*p_e
        return C1*C2 + C3

    def _Isp(self, F, mdot_e):
        return F/(self.g_0*mdot_e)

    def _Isp_mean(self, F_vac, t, mdot_ox_T, mdot_fuel_T):
        num = np.trapz(F_vac, x=t)
        denom = self.g_0*(mdot_ox_T + mdot_fuel_T)
        return num/denom

    def _dP0_dt(self, x, total):
        P_0, R_c, M_LOX, M_HTPB = x
        A_burn = self._A_burn(R_c)
        O_F = M_LOX/M_HTPB
        rdot_fuel = self._port_regression_rate(P_0, R_c, O_F, total)
        V_c = self._V_chamber(R_c)
        rho_fuel = self.rho_fuel
        R_g = self.f_Rg_of(O_F)[()]
        T_0 = self.f_T0_of(O_F)[()]*self.eta_t
        A_star = self.A_star
        gamma = self.f_gamma_of(O_F)[()]
        mdot_ox = self._mdot_ox(P_0)
        C1 = A_burn*rdot_fuel*(rho_fuel*R_g*T_0 - P_0)/V_c
        C2_E1 = (gamma + 1.)/(gamma - 1.)
        C2_1 = np.sqrt(gamma*R_g*T_0*(2./(gamma + 1.))**C2_E1)
        C2 = P_0*A_star*C2_1/V_c
        C3 = R_g*T_0*mdot_ox/V_c
        return C1 - C2 + C3

    def _f_x(self, x, t, total, verbose=False):
        P_0, R_c, M_LOX, M_HTPB = x
        if verbose:
            print(f"t={t:1.4f}", f"P0: {P_0:1.4f}", f"R_c: {R_c:1.4f}", f"M_LOX: {M_LOX:1.4f}", f"M_HTPB: {M_HTPB:1.4f}")
        O_F = M_LOX/M_HTPB
        dP0_dt = self._dP0_dt(x, total)
        dr_dt = self._port_regression_rate(P_0, R_c, O_F, total)
        dmox_dt = self._mdot_ox(P_0)
        dmfuel_dt = self._mdot_fuel(R_c, dr_dt)
        xdot = np.array([dP0_dt, dr_dt, -dmox_dt, -dmfuel_dt])
        if verbose:
            print(f"\tdP0_dt: {dP0_dt:1.4f}", f"dRc_dt: {dr_dt:1.4f}", f"mdot_ox: {dmox_dt:1.4f}", f"mdot_fuel: {dmfuel_dt:1.4f}")
        return xdot

    def hybrid_burn_analysis(self, t_burn, M_fuel_0, O_F, dt_steady=2e-3, dt_trans=1.5e-3, total=False):
        t_trans = np.arange(0., 0.1 + dt_trans, dt_trans)
        t_steady = np.arange(0.1 + dt_steady, t_burn + dt_steady, dt_steady)
        t = np.concatenate((t_trans, t_steady))
        self.rdot_t = np.zeros_like(t)
        self.M_OF_t = np.zeros_like(t)
        self.F_Tvac_t = np.zeros_like(t)
        self.F_Tsl_t = np.zeros_like(t)
        self.p_inj_t = np.zeros_like(t)
        self.P_0_t = np.zeros_like(t)
        self.R_c_t = np.zeros_like(t)
        self.T_0_theory_t = np.zeros_like(t)
        self.T_0_actual_t = np.zeros_like(t)
        self.mdot_ox_t = np.zeros_like(t)
        self.mdot_fuel_t = np.zeros_like(t)
        self.mdot_tot_t = np.zeros_like(t)
        self.mdot_choke_t = np.zeros_like(t)
        self.Isp_accum_t = np.zeros_like(t)
        self.m_remain_ox_t = np.zeros_like(t)
        self.m_remain_fuel_t = np.zeros_like(t)
        self.m_remain_tot_t = np.zeros_like(t)
        self.Ispvac_t = np.zeros_like(t)
        self.Ispsl_t = np.zeros_like(t)
        self.G_ox_t = np.zeros_like(t)
        self.cstar_theory_t = np.zeros_like(t)
        self.cstar_actual_t = np.zeros_like(t)
        M_ox_0 = O_F*M_fuel_0
        O_F = M_ox_0/M_fuel_0
        x0 = np.array([self.p_amb*2., self.D_c_t0/2., M_ox_0, M_fuel_0])
        x_t = np.zeros((len(t), 4))
        xdot_t = np.zeros((len(t), 4))
        x = x0
        for k in range(len(t)):
            if (x[2] <= 0.)*(x[3] <= 0.):
                break
            if t[k] < 0.1:
                self.dt = dt_trans
            else:
                self.dt = dt_steady
            self.M_OF_t[k] = O_F
            self.mdot_choke_t[k] = self._mdot_exit(x[0], O_F)
            self.T_0_theory_t[k] = self.f_T0_of(O_F)[()]
            self.T_0_actual_t[k] = self.T_0_theory_t[k]*self.eta_t
            self.P_0_t[k] = x[0]
            self.p_inj_t[k] = self.p_inj
            self.R_c_t[k] = x[1]
            self.m_remain_ox_t[k] = x[2]
            self.m_remain_fuel_t[k] = x[3]
            self.m_remain_tot_t[k] = x[2] + x[3]
            self.mdot_ox_t[k] = self._mdot_ox(x[0])
            self.rdot_t[k] = self._port_regression_rate(x[0], x[1], O_F, total)
            self.mdot_fuel_t[k] = self._mdot_fuel(x[1], self.rdot_t[k])
            self.mdot_tot_t[k] = self.mdot_ox_t[k] + self.mdot_fuel_t[k]
            self.mdot_choke_t[k] = self._mdot_exit(x[0], O_F)
            self.F_Tvac_t[k] = self._vacthrust_eq(x[0], O_F)
            self.F_Tsl_t[k] = self._slthrust_eq(self.F_Tvac_t[k])
            self.Isp_accum_t[k] = self._Isp_mean(self.F_Tvac_t, t, self.mdot_ox_t[k], self.mdot_fuel_t[k])
            self.Ispvac_t[k] = self._Isp(self.F_Tvac_t[k], self.mdot_choke_t[k])
            self.Ispsl_t[k] = self._Isp(self.F_Tsl_t[k], self.mdot_choke_t[k])
            self.G_ox_t[k] = self.mdot_ox_t[k]/self._A_chamber(x[1])
            self.cstar_theory_t[k] = self._c_star(O_F, self.T_0_actual_t[k])
            self.cstar_actual_t[k] = x[0]*self.A_star/self.mdot_tot_t[k]
            O_F = self.mdot_ox_t[k]/self.mdot_fuel_t[k]
            f_x = self._f_x(x, t[k], total, verbose=True)
            xdot_t[k, :] = f_x
            xhatp1 = x + self.dt*f_x
            f_xhatp1 = self._f_x(xhatp1, t[k], total=total)
            xp1 = x + 0.5*self.dt*(f_x + f_xhatp1)
            x = xp1
            xdot = self._f_x(x, t[k], total)
        if total:
            str_label = 'total'
        else:
            str_label = 'classic'
        thrust_fig, thrust_axis = plt.subplots()
        thrust_axis.semilogx(t, self.F_Tsl_t/1000., label=r'$T_\mathrm{sl}$', color='k')
        thrust_axis.semilogx(t, self.F_Tvac_t/1000., label=r'$T_\mathrm{vac}$', color='k', linestyle=':')
        thrust_axis.set_ylim(0, 12)
        thrust_axis.set_xlim(1e-3, 6)
        thrust_axis.set_xlabel("Time, s")
        thrust_axis.set_ylabel("Thrust, kN")
        thrust_axis.legend()
        thrust_fig.savefig('./Figures/Thrust_' + str_label + '.pdf')

        pressure_fig, pressure_axis = plt.subplots()
        pressure_axis.semilogx(t, self.P_0_t/1000., label=r'$P_0$', color='k')
        pressure_axis.semilogx(t, self.p_inj_t/1000., label=r'$p_\mathrm{inj}$', color='k', linestyle=':')
        pressure_axis.set_ylim(1000,)
        pressure_axis.set_xlim(1e-3, 6)
        pressure_axis.set_xlabel("Time, s")
        pressure_axis.set_ylabel("Pressure, kPa")
        pressure_axis.legend()
        pressure_fig.savefig('./Figures/Pressure_' + str_label + '.pdf')

        massflow_fig, massflow_axis = plt.subplots()
        massflow_axis.semilogx(t, self.mdot_ox_t, label=r'$\dot{m}_\mathrm{ox}$', color='k', linestyle='-.')
        massflow_axis.semilogx(t, self.mdot_fuel_t, label=r'$\dot{m}_\mathrm{fuel}$', color='k', linestyle='--')
        massflow_axis.semilogx(t, self.mdot_tot_t, label=r'$\dot{m}_\mathrm{total}$', color='k')
        massflow_axis.semilogx(t, self.mdot_choke_t, label=r'$\dot{m}_\mathrm{choke}$', color='k', linestyle=':')
        massflow_axis.set_ylim(0,9)
        massflow_axis.set_xlim(1e-3, 6)
        massflow_axis.set_xlabel("Time, s")
        massflow_axis.set_ylabel("Mass Flow, kg/s")
        massflow_axis.legend()
        massflow_fig.savefig('./Figures/MassFlow_' + str_label + '.pdf')

        OF_fig, OF_axis = plt.subplots()
        OF_axis.semilogx(t, self.M_OF_t, color='r')
        rdot_axis = OF_axis.twinx()
        rdot_axis.semilogx(t, self.rdot_t*100., color='b')
        rdot_axis.set_xlim(1e-3, 6)
        OF_axis.set_ylabel("O/F Ratio", color='r')
        rdot_axis.set_ylabel("Regression Rate, cm/s", color='b')
        OF_axis.set_xlabel("Time, s")
        OF_fig.savefig('./Figures/Regression_' + str_label + '.pdf')

        Isp_fig, Isp_axis = plt.subplots()
        Isp_axis.semilogx(t, self.Ispvac_t, label=r'Vacuum $I_\mathrm{sp}$', color='k', linestyle='-')
        Isp_axis.semilogx(t, self.Ispsl_t, label=r'Sea Level $I_\mathrm{sp}$', color='k', linestyle='--')
        Isp_axis.semilogx(t, self.Isp_accum_t, label=r'Accumulated $I_\mathrm{sp}$', color='k', linestyle=':')
        Isp_axis.set_ylim(0, 500)
        Isp_axis.set_xlim(1e-3, 6)
        Isp_axis.set_xlabel("Time, s")
        Isp_axis.set_ylabel(r'$I_\mathrm{sp}$, s')
        Isp_axis.legend()
        Isp_fig.savefig('./Figures/Isp_' + str_label + '.pdf')

        massrem_fig, massrem_axis = plt.subplots()
        massrem_axis.semilogx(t, self.m_remain_ox_t, label=r'Oxidizer Mass', color='k', linestyle='-.')
        massrem_axis.semilogx(t, self.m_remain_fuel_t, label=r'Fuel Mass', color='k', linestyle='--')
        massrem_axis.semilogx(t, self.m_remain_tot_t, label=r'Total Mass', color='k')
        massrem_axis.set_ylim(0, )
        massrem_axis.set_xlim(1e-3, 6)
        massrem_axis.set_xlabel("Time, s")
        massrem_axis.set_ylabel(r'Remaining Mass, kg')
        massrem_axis.legend()
        massrem_fig.savefig('./Figures/Mass_Remaining_' + str_label + '.pdf')

        regression_fig, regression_axis = plt.subplots()
        regression_axis.scatter(self.G_ox_t*100**2, self.rdot_t*100., label=r'Simulation Data', edgecolor='k', facecolor='None')
        eq_fit = lambda x: x[0]*(self.G_ox_t*100**2)**x[1]*self.L_P**x[2] - self.rdot_t*100.
        a, n, m = optimize.leastsq(eq_fit, [1., 0.8, 1.])[0]
        regression_axis.plot(self.G_ox_t*100**2, a*(self.G_ox_t*100**2)**n*self.L_P**m, label=r'$\dot{{r}} = {:.4g}G_\mathrm{{ox}}^{{{:.4f}}}L^{{{:.2f}}}$'.format(a, n, m))
        regression_axis.set_xlabel(r'$G_\mathrm{ox}$, $\mathrm{kg/(s\cdot cm^2)}$')
        regression_axis.set_ylabel(r'Regression Rate, cm/s')
        regression_axis.legend()
        regression_fig.savefig('./Figures/Rdot_Gox_' + str_label + '.pdf')

        cstar_fig, cstar_axis = plt.subplots()
        cstar_axis.plot(self.M_OF_t, self.cstar_theory_t, label='Theoretical', color='k', linestyle=':')
        cstar_axis.plot(self.M_OF_t, self.cstar_actual_t, label='Actual', color='k')
        cstar_axis.set_xlabel("O/F Ratio")
        cstar_axis.set_ylabel(r'$c^*$, m/s')
        cstar_axis.legend()
        cstar_fig.savefig('./Figures/Cstar_' + str_label + '.pdf')


if __name__ == "__main__":
    plt.close('all')
    case = incompressible_hybrid(3000.0e3)
    case.hybrid_burn_analysis(6., 2.95, 6.0)

    plt.close('all')
    case_total = incompressible_hybrid(3100.0e3)
    case_total.hybrid_burn_analysis(6., 3.25, 6.0, total=True)
