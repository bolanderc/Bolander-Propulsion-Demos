#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:12:29 2021

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
from scipy.interpolate import RegularGridInterpolator as rgi

class compressible_hybrid:
    def __init__(self, P_inj, P_c):
        self.exp_ratio = 2.25
        self.D_t = 1.0e-2  # m
        self.A_star = np.pi*self.D_t*self.D_t/4.
        self.A_e = self.exp_ratio*self.A_star
        self.D_e = np.sqrt(4.*self.A_e/np.pi)  # m
        self.theta_e = np.deg2rad(15.)  # rad
        self.D_c_t0 = 1.5e-2  # m
        self.L_P = 31.5e-2  # m
        self.p_amb = 60.0e3  # Pa
        self.p_sl = 101.325e3  # Pa
        self.p_inj = P_inj  # Pa
        self.g_0 = 9.8067  # m/s^2
        self.eps_combustor = 0.9
        self.rho_fuel = 975.  # kg/m^3
        self.h_v_fuel_surface = 3.0e6  # J/kg
        self.T_fuel_surface = 293.15  # K
        self.Pc_mean = P_c  # Pa

        self.N_P_inj = 1
        self.D_P_inj = 0.25e-2  # m
        # self.D_P_inj = 0.125e-2
        self.A_inj = self.N_P_inj*np.pi*self.D_P_inj*self.D_P_inj/4.  # m^2
        self.D_GOX_feed = 0.6325e-2  # m
        # self.D_GOX_feed = 0.5e-2
        self.A_GOX_feed = np.pi*self.D_GOX_feed**2/4.  # m^2

        self.beta_ratio = self.D_P_inj/self.D_GOX_feed
        self.C_vinc = 1.
        self.C_dcomp = 1.

        data_dir = './GOX_CEA_Table_Data'
        Pc_range = np.load(data_dir + '/Pc_data.npy')*1000.  # Pa
        OF_range = np.load(data_dir + '/OF_data.npy')
        cstar_data = np.load(data_dir + '/Cstar_data.npy')
        gamma_data = np.load(data_dir + '/Gamma_data.npy')
        MW_data = np.load(data_dir + '/MW_data.npy')  # kg/kg-mol
        Pr_data = np.load(data_dir + '/Pr_data.npy')
        Tflame_data = np.load(data_dir + '/Tflame_data.npy')  # K
        visc_data = np.load(data_dir + '/Visc_data.npy')  # Pa-s
        self.R_u = 8314.  # J/(kg mol K)
        Rg_data = self.R_u/MW_data  # J/(kg K)
        Cp_data = np.divide(np.multiply(gamma_data, Rg_data), gamma_data - 1.)  # J/(kg K)
        self.eta_t = 0.9**2
        self.f_cstar_of = rgi((Pc_range, OF_range), cstar_data, bounds_error=False, fill_value=None)
        self.f_gamma_of = rgi((Pc_range, OF_range), gamma_data, bounds_error=False, fill_value=None)
        self.f_MW_of = rgi((Pc_range, OF_range), MW_data, bounds_error=False, fill_value=None)
        # self.f_Pr_of = rgi((Pc_range, OF_range), Pr_data, bounds_error=False, fill_value=None)
        self.f_Pr_of = 0.5
        self.f_T0_of = rgi((Pc_range, OF_range), Tflame_data, bounds_error=False, fill_value=None)
        self.f_mu_of = rgi((Pc_range, OF_range), visc_data, bounds_error=False, fill_value=None)
        self.f_Rg_of = rgi((Pc_range, OF_range), Rg_data, bounds_error=False, fill_value=None)
        self.f_Cp_of = rgi((Pc_range, OF_range), Cp_data, bounds_error=False, fill_value=None)

    def _southerland(self, O_F):
        T_0 = self.T_inj
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

    def _Cv(self, Re_D):
        beta = self.beta_ratio
        C1 = 0.598 + 0.468*(beta**4 + 10*beta**8)
        C2 = np.sqrt(1. - beta**4)
        C3 = (0.87 + 8.1*beta**4)
        C4 = C2/np.sqrt(Re_D)
        C_v = C1*C2 + C3*C4
        return C_v

    def _Cd(self, C_v):
        return C_v/np.sqrt(1. - self.beta_ratio**4)

    def _P0_true(self, p1, p2, gamma):
        A1 = self.A_GOX_feed
        A2 = self.A_inj
        area_ratio = A1/A2
        num_exp = (gamma + 1.)/gamma
        denom_exp = 2./gamma
        total_exp = gamma/(gamma - 1.)
        num = area_ratio**2*p1**num_exp - p2**num_exp
        denom = area_ratio**2*p1**denom_exp - p2**denom_exp
        P0 = (num/denom)**total_exp
        return P0

    def _incompressible_Cd(self, C_v0, P_0, O_F):
        error = 100.
        P1 = self.p_inj
        P2 = P_0
        D1 = self.D_GOX_feed
        R_g = self.Rg_inj
        T1 = self.T_inj
        mu = self._southerland(O_F)
        # R_g = self.R_u/32.
        # T1 = 293.15
        # mu = 2.02299e-5
        rho_inj = self.rho_inj
        C_d0 = self._Cd(C_v0)
        while error > 1e-7:
            mdot = self._mdot_incompressible(C_v0, rho_inj, P1, P2)
            Re_D1 = 4.*mdot/(np.pi*D1*mu)
            C_v = self._Cv(Re_D1)
            C_d = self._Cd(C_v)
            error = np.abs((C_d - C_d0)/C_d0)
            C_d0 = C_d
            C_v0 = C_v
        return C_d

    def _compressible_Cd(self, f, K_n, gamma, p_ratio):
        test = ((gamma + 1.)/2.)**(gamma/(gamma - 1.))
        if p_ratio < test:
            r = 1./p_ratio
            C1 = (2.*r**(1./gamma))**2
            C2 = (1. - r)/K_n**2
            num = 1. - np.sqrt(1. - f*C1*C2)
            denom = 2.*f*r**(1./gamma)
            C_dcomp = num/denom
            return C_dcomp
        else:
            r = (2./(gamma + 1.))**(gamma/(gamma - 1.))
            p_ratio = 1./p_ratio
            C1 = r - p_ratio
            C2 = r**(1./gamma)
            C3 = (2.*C2)**2
            C4 = f*(1. - p_ratio)
            C5 = 1./(2.*f*C2)
            C_dcomp = C5*((1. + C1*C2/K_n**2) - np.sqrt((1. + C1*C2/K_n**2)**2 - C3*C4/K_n**2))
            return C_dcomp


    def _mdot_incompressible(self, C_v, rho, p1, p2):
        Cd = self._Cd(C_v)
        C1 = Cd*self.A_inj
        C2 = np.sqrt(2.*rho*(p1 - p2))
        return C1*C2


    def _Kn(self, p_ratio, gamma):
        test = ((gamma + 1.)/2.)**(gamma/(gamma - 1.))
        if p_ratio < test:
            p_ratio = 1./p_ratio
            C1 = 2.*gamma/(gamma - 1.)
            C2 = p_ratio**(2./gamma)
            C3 = 1. - p_ratio**((gamma - 1.)/gamma)
            return np.sqrt(C1*C2*C3)
        else:
            C1 = 2./(gamma + 1.)
            E1 = (gamma + 1.)/(gamma - 1.)
            return np.sqrt(gamma*C1**E1)

    def _compressible_corrections(self, P_c, O_F):
        gamma = 1.4
        self.P_0inj = self._P0_true(self.p_inj, P_c, gamma)
        self.Rg_inj = self.R_u/32.  # J/kg-K
        self.T_inj = 293.15  # K
        self.rho_inj = self.P_0inj/(self.Rg_inj*self.T_inj)
        self.C_dinc = self._incompressible_Cd(self.C_vinc, P_c, O_F)
        f = 1./self.C_dinc - 1./(2.*self.C_dinc**2)
        self.K_n = self._Kn(self.P_0inj/P_c, gamma)
        self.C_dcomp = self._compressible_Cd(f, self.K_n, gamma, self.P_0inj/P_c)
        # self.C_dcomp = 0.82

    def _mdot_ox(self, P_0, O_F, comp_correction):
        if comp_correction:
            gamma = 1.4
            # test = ((gamma + 1.)/2.)**(gamma/(gamma - 1.))
            # if self.P_0inj/P_0 < test:
            #     r = P_0/self.P_0inj
            #     A = 2.*gamma*self.rho_inj*self.P_0inj/(gamma - 1.)
            #     B = r**(2./gamma)
            #     C = r**((gamma + 1.)/gamma)
            #     mdot_ox = self.K_n*self.C_dcomp*self.A_inj*np.sqrt(A*(B - C))
            # else:
            #     A = gamma*self.rho_inj*self.P_0inj
            #     B = (2./(gamma + 1.))**((gamma + 1.)/(gamma - 1.))
            #     mdot_ox = self.K_n*self.C_dcomp*self.A_inj*np.sqrt(A*B)
            mdot_ox = self.K_n*self.C_dcomp*self.A_inj*self.P_0inj/np.sqrt(self.Rg_inj*self.T_inj)
        else:
            mdot_ox = self.C_dinc*self.A_inj*np.sqrt(2.*self.rho_inj*(self.P_0inj - P_0))
        return mdot_ox


    def _mdot_fuel(self, R_c, rdot_fuel):
        A_burn = self._A_burn(R_c)
        return self.rho_fuel*A_burn*rdot_fuel

    def _G_ox(self, P_0, R_c, rdot_fuel):
        mdot_ox = self._mdot_ox
        mdot_fuel = self._mdot_fuel(R_c, rdot_fuel)
        G_ox = mdot_ox/mdot_fuel
        return G_ox

    def _dh_flame(self, P_0, O_F):
        T_flame = self.f_T0_of([P_0, O_F])[0]
        C_p_fuel = self.f_Cp_of([P_0, O_F])[0]
        dh_flame = C_p_fuel*(T_flame - self.T_fuel_surface)
        return dh_flame

    def _D_c_t(self, t):
        rdot_int = np.trapz(self.rdot_t, dx=self.dt)
        D_c_t = self.D_c_t0 + 2.*rdot_int
        return D_c_t

    def _port_regression_rate(self, P_0, R_c, O_F, total, comp_correction):
        mu_fuel = self.f_mu_of([P_0, O_F])[0]
        # Pr = self.f_Pr_of([P_0, O_F])[0]
        Pr = self.f_Pr_of
        C1 = 0.047/(Pr**(2./3.)*self.rho_fuel)
        dh_flame = self._dh_flame(P_0, O_F)
        C2 = dh_flame/self.h_v_fuel_surface
        E1 = 0.23
        mdot_ox = self._mdot_ox(P_0, O_F, comp_correction)
        A_c = self._A_chamber(R_c)
        E2 = 4./5.
        C4 = mu_fuel/self.L_P
        E3 = 1./5.
        if not total:
            C3 = mdot_ox/A_c
            return C1*(C2**E1)*(C3**E2)*(C4**E3)
        else:
            C3_1 = (mdot_ox/A_c)**E3
            C3_2 = (5./9.)*(0.047/(Pr)**E1)
            C3_3 = C2**E1
            C3_4 = C4**E3
            C3_5 = self.L_P/(2.*R_c)
            C3 = (C3_1 + C3_2*C3_3*C3_4*C3_5)
            E4 = 4.
            return C1*(C2**E1)*(C3**E4)*(C4**E3)

    def _mdot_exit(self, P_0, O_F):
        gamma = self.f_gamma_of([P_0, O_F])[0]
        T_0 = self.f_T0_of([P_0, O_F])[0]
        R_g = self.f_Rg_of([P_0, O_F])[0]
        A_star = self.A_star
        C1 = P_0*A_star
        C2 = gamma/(R_g*T_0)
        C3 = (2./(gamma + 1.))
        E1 = (gamma + 1.)/(gamma - 1.)
        return C1*np.sqrt(C2*C3**E1)

    def _c_star(self, P_0, O_F, T_0):
        gamma = self.f_gamma_of([P_0, O_F])[0]
        R_g = self.f_Rg_of([P_0, O_F])[0]
        C1 = np.sqrt(gamma*R_g*T_0)
        C2 = (2./(gamma + 1.))
        E1 = (gamma + 1.)/(gamma - 1.)
        return C1/(gamma*np.sqrt(C2**E1))

    def _slthrust_eq(self, F_vac):
        return F_vac - self.A_e*self.p_sl

    def _thrust_eq(self, P_0, O_F, p_inf=0.):
        mdot_e = self._mdot_exit(P_0, O_F)
        gamma = self.f_gamma_of([P_0, O_F])[0]
        T_0_actual = self.f_T0_of([P_0, O_F])[0]*self.eta_t
        R_g = self.f_Rg_of([P_0, O_F])[0]
        M_e = comp_codes.area_mach(1.4, A_Astar=self.exp_ratio, gamma=gamma)
        p_e = comp_codes.isenflowprops("P", M_e, gamma=gamma, stagnation=P_0)
        T_e = comp_codes.isenflowprops("T", M_e, gamma=gamma, stagnation=T_0_actual)
        lam_t = 0.5*(1. + np.cos(self.theta_e))
        C1 = lam_t*mdot_e*M_e
        C2 = np.sqrt(gamma*R_g*T_e)
        C3 = self.A_e*(p_e - p_inf)
        return C1*C2 + C3

    def _Isp(self, F, mdot_e):
        return F/(self.g_0*mdot_e)

    def _Isp_mean(self, F_vac, t, M_OX_T, M_HTPB_T):
        num = np.trapz(F_vac, x=t)
        denom = self.g_0*(M_OX_T + M_HTPB_T)
        return num/denom

    def _dP0_dt(self, x, total, comp_correction):
        P_0, R_c, M_LOX, M_HTPB = x
        A_burn = self._A_burn(R_c)
        O_F = M_LOX/M_HTPB
        rdot_fuel = self._port_regression_rate(P_0, R_c, O_F, total, comp_correction)
        V_c = self._V_chamber(R_c)
        rho_fuel = self.rho_fuel
        R_g = self.f_Rg_of([P_0, O_F])[0]
        T_0 = self.f_T0_of([P_0, O_F])[0]*self.eta_t
        A_star = self.A_star
        gamma = self.f_gamma_of([P_0, O_F])[0]
        mdot_ox = self._mdot_ox(P_0, O_F, comp_correction)
        C1 = A_burn*rdot_fuel*(rho_fuel*R_g*T_0 - P_0)/V_c
        C2_E1 = (gamma + 1.)/(gamma - 1.)
        C2_1 = np.sqrt(gamma*R_g*T_0*(2./(gamma + 1.))**C2_E1)
        C2 = P_0*A_star*C2_1/V_c
        C3 = R_g*T_0*mdot_ox/V_c
        return C1 - C2 + C3

    def _f_x(self, x, t, total, comp_correction, verbose=False):
        P_0, R_c, M_LOX, M_HTPB = x
        if verbose:
            print(f"t={t:1.4f}", f"P0: {P_0:1.4f}", f"R_c: {R_c:1.4f}", f"M_LOX: {M_LOX:1.4f}", f"M_HTPB: {M_HTPB:1.4f}")
        O_F = M_LOX/M_HTPB
        dP0_dt = self._dP0_dt(x, total, comp_correction)
        dr_dt = self._port_regression_rate(P_0, R_c, O_F, total, comp_correction)
        dmox_dt = self._mdot_ox(P_0, O_F, comp_correction)
        dmfuel_dt = self._mdot_fuel(R_c, dr_dt)
        xdot = np.array([dP0_dt, dr_dt, -dmox_dt, -dmfuel_dt])
        if verbose:
            print(f"\tdP0_dt: {dP0_dt:1.4f}", f"dRc_dt: {dr_dt:1.4f}", f"mdot_ox: {dmox_dt:1.4f}", f"mdot_fuel: {dmfuel_dt:1.4f}")
        return xdot

    def hybrid_burn_analysis(self, t_burn, M_fuel_0, dt_steady=2.0e-3, dt_trans=0.1e-3, total=True, comp_correction=False):
        transition = 0.45
        t_trans = np.arange(0., transition + dt_trans, dt_trans)
        t_steady = np.arange(transition + dt_steady, t_burn + dt_steady, dt_steady)
        t = np.concatenate((t_trans, t_steady))
        self.rdot_t = np.zeros_like(t)
        self.M_OF_t = np.zeros_like(t)
        self.F_Tvac_t = np.zeros_like(t)
        self.F_Tsl_t = np.zeros_like(t)
        self.F_T_t = np.zeros_like(t)
        self.p_inj_t = np.zeros_like(t)
        self.P_0_t = np.zeros_like(t)
        self.R_c_t = np.zeros_like(t)
        self.T_0_theory_t = np.zeros_like(t)
        self.T_0_actual_t = np.zeros_like(t)
        self.mdot_ox_t = np.zeros_like(t)
        self.mdot_fuel_t = np.zeros_like(t)
        self.mdot_tot_t = np.zeros_like(t)
        self.mdot_choke_t = np.zeros_like(t)
        self.Isp_t = np.zeros_like(t)
        self.m_remain_ox_t = np.zeros_like(t)
        self.m_remain_fuel_t = np.zeros_like(t)
        self.m_remain_tot_t = np.zeros_like(t)
        self.Ispvac_t = np.zeros_like(t)
        self.Ispsl_t = np.zeros_like(t)
        self.G_ox_t = np.zeros_like(t)
        self.cstar_theory_t = np.zeros_like(t)
        self.cstar_actual_t = np.zeros_like(t)
        M_ox_0 = 1.4*M_fuel_0
        O_F = M_ox_0/M_fuel_0
        x0 = np.array([self.Pc_mean, self.D_c_t0/2., M_ox_0, M_fuel_0])
        xdot_t = np.zeros((len(t), 4))
        x = x0
        if comp_correction:
            self._compressible_corrections(self.Pc_mean, O_F)
        else:
            gamma = 1.4
            self.P_0inj = self._P0_true(self.p_inj, self.Pc_mean, gamma)
            self.Rg_inj = self.R_u/32.
            self.T_inj = 293.15
            self.rho_inj = self.P_0inj/(self.Rg_inj*self.T_inj)
            self.C_dinc = self._incompressible_Cd(1.0, self.Pc_mean, O_F)
        for k in range(len(t)):
            if (x[2] <= 0.)*(x[3] <= 0.):
                break
            if t[k] < transition:
                self.dt = dt_trans
            else:
                self.dt = dt_steady
            self.M_OF_t[k] = O_F
            self.mdot_choke_t[k] = self._mdot_exit(x[0], O_F)
            self.T_0_theory_t[k] = self.f_T0_of([x[0], O_F])[0]
            self.T_0_actual_t[k] = self.T_0_theory_t[k]*self.eta_t
            self.P_0_t[k] = x[0]
            self.p_inj_t[k] = self.p_inj
            self.R_c_t[k] = x[1]
            self.m_remain_ox_t[k] = x[2]
            self.m_remain_fuel_t[k] = x[3]
            self.m_remain_tot_t[k] = x[2] + x[3]
            self.mdot_ox_t[k] = self._mdot_ox(x[0], O_F, comp_correction)
            self.rdot_t[k] = self._port_regression_rate(x[0], x[1], O_F, total, comp_correction)
            self.mdot_fuel_t[k] = self._mdot_fuel(x[1], self.rdot_t[k])
            self.mdot_tot_t[k] = self.mdot_ox_t[k] + self.mdot_fuel_t[k]
            self.F_Tvac_t[k] = self._thrust_eq(x[0], O_F)
            self.F_T_t[k] = self._thrust_eq(x[0], O_F, p_inf=self.p_amb)
            self.F_Tsl_t[k] = self._slthrust_eq(self.F_Tvac_t[k])
            self.Isp_t[k] = self._Isp(self.F_T_t[k], self.mdot_choke_t[k])
            self.Ispvac_t[k] = self._Isp(self.F_Tvac_t[k], self.mdot_choke_t[k])
            self.Ispsl_t[k] = self._Isp(self.F_Tsl_t[k], self.mdot_choke_t[k])
            O_F = self.mdot_ox_t[k]/self.mdot_fuel_t[k]
            f_x = self._f_x(x, t[k], total, comp_correction, verbose=False)
            xdot_t[k, :] = f_x
            xhatp1 = x + self.dt*f_x
            f_xhatp1 = self._f_x(xhatp1, t[k], total, comp_correction)
            xp1 = x + 0.5*self.dt*(f_x + f_xhatp1)
            x = xp1
        if total:
            str_label = 'total'
        else:
            str_label = 'classic'
        if comp_correction:
            color='r'
        else:
            color='k'
            self.thrust_fig, self.thrust_axis = plt.subplots()
            self.pressure_fig, self.pressure_axis = plt.subplots()
            self.massflow_fig, self.massflow_axis = plt.subplots()
            self.OF_fig, self.OF_axis = plt.subplots()
            self.massrem_fig, self.massrem_axis = plt.subplots()
            self.pressure_ratio_fig, self.pressure_ratio_axis = plt.subplots()
            self.Isp_fig, self.Isp_axis = plt.subplots()

        self.total_impulse = np.sum(self.F_T_t[:len(t_trans)])*dt_trans + np.sum(self.F_T_t[len(t_trans):])*dt_steady
        self.mean_thrust = np.average(self.F_T_t)
        self.mean_Isp = np.average(self.Isp_t)
        self.mean_injector_ratio = np.average(self.P_0inj/self.P_0_t)
        self.m_ox_consumed = M_ox_0 - self.m_remain_ox_t[-1]
        self.m_fuel_consumed = M_fuel_0 - self.m_remain_fuel_t[-1]
        self.m_total_consumed = self.m_ox_consumed + self.m_fuel_consumed
        self.thrust_axis.plot(t, self.F_Tsl_t/1000., label=r'$T_\mathrm{sl}$', color=color)
        self.thrust_axis.plot(t, self.F_Tvac_t/1000., label=r'$T_\mathrm{vac}$', color=color, linestyle=':')
        self.thrust_axis.plot(t, self.F_T_t/1000., label=r'$T_d$', color=color, linestyle='-.')
        # self.thrust_axis.set_ylim(0, 12)
        self.thrust_axis.set_xlim(0, t_burn)
        self.thrust_axis.set_xlabel("Time, s")
        self.thrust_axis.set_ylabel("Thrust, kN")
        if not comp_correction:
            self.thrust_axis.legend()
        else:
            self.thrust_fig.savefig('./Figures/Thrust_' + str_label + '.pdf')

        self.pressure_axis.plot(t, self.P_0_t/1000., label=r'$P_0$', color=color)
        self.pressure_axis.plot(t, self.p_inj_t/1000., label=r'$p_\mathrm{inj}$', color=color, linestyle=':')
        # self.pressure_axis.set_ylim(1000,)
        self.pressure_axis.set_xlim(0, t_burn)
        self.pressure_axis.set_xlabel("Time, s")
        self.pressure_axis.set_ylabel("Pressure, kPa")
        if not comp_correction:
            self.pressure_axis.legend()
        else:
            self.pressure_fig.savefig('./Figures/Pressure_' + str_label + '.pdf')

        self.massflow_axis.plot(t, self.mdot_ox_t*1000., label=r'$\dot{m}_\mathrm{ox}$', color=color, linestyle='-.')
        self.massflow_axis.plot(t, self.mdot_fuel_t*1000., label=r'$\dot{m}_\mathrm{fuel}$', color=color, linestyle='--')
        self.massflow_axis.plot(t, self.mdot_tot_t*1000., label=r'$\dot{m}_\mathrm{total}$', color=color)
        self.massflow_axis.plot(t, self.mdot_choke_t*1000., label=r'$\dot{m}_\mathrm{choke}$', color=color, linestyle=':')
        # self.massflow_axis.set_ylim(0,9)
        self.massflow_axis.set_xlim(0, t_burn)
        self.massflow_axis.set_xlabel("Time, s")
        self.massflow_axis.set_ylabel("Mass Flow, kg/s")
        if not comp_correction:
            self.massflow_axis.legend()
        else:
            self.massflow_fig.savefig('./Figures/MassFlow_' + str_label + '.pdf')

        self.OF_axis.plot(t, self.M_OF_t, color=color)
        self.OF_axis.set_xlim(0, t_burn)
        self.OF_axis.set_ylabel("O/F Ratio")
        self.OF_axis.set_xlabel("Time, s")
        self.OF_fig.savefig('./Figures/OF_' + str_label + '.pdf')

        self.massrem_axis.plot(t, self.m_remain_ox_t, label=r'Oxidizer Mass', color=color, linestyle='-.')
        self.massrem_axis.plot(t, self.m_remain_fuel_t, label=r'Fuel Mass', color=color, linestyle='--')
        self.massrem_axis.plot(t, self.m_remain_tot_t, label=r'Total Mass', color=color)
        # self.massrem_axis.set_ylim(0, )
        self.massrem_axis.set_xlim(0, t_burn)
        self.massrem_axis.set_xlabel("Time, s")
        self.massrem_axis.set_ylabel(r'Remaining Mass, kg')
        if not comp_correction:
            self.massrem_axis.legend()
        else:
            self.massrem_fig.savefig('./Figures/Mass_Remaining_' + str_label + '.pdf')

        self.pressure_ratio_axis.plot(t, self.P_0inj/self.P_0_t, color=color)
        self.pressure_ratio_axis.set_xlim(0, t_burn)
        self.pressure_ratio_axis.set_xlabel("Time, s")
        self.pressure_ratio_axis.set_ylabel(r'$P_{0,\mathrm{inj}}/P_{0_c}$')

        self.Isp_axis.plot(t, self.Ispvac_t, label=r'Vacuum $I_\mathrm{sp}$', color=color, linestyle='-')
        self.Isp_axis.plot(t, self.Ispsl_t, label=r'Sea Level $I_\mathrm{sp}$', color=color, linestyle='--')
        self.Isp_axis.plot(t, self.Isp_t, label=r'Design $I_\mathrm{sp}$', color=color, linestyle='-.')
        self.Isp_axis.set_ylim(0, 500)
        self.Isp_axis.set_xlim(0, t_burn)
        self.Isp_axis.set_xlabel("Time, s")
        self.Isp_axis.set_ylabel(r'$I_\mathrm{sp}$, s')
        if not comp_correction:
            self.Isp_axis.legend()
        else:
            self.Isp_fig.savefig('./Figures/Isp_' + str_label + '.pdf')



if __name__ == "__main__":
    plt.close('all')
    t_burn = 20.
    case_total = compressible_hybrid(2500.0e3, 793.38e3)
    case_total.hybrid_burn_analysis(t_burn, 1., comp_correction=False)
    print("Incompressible Analysis:")
    print(f"C_d = {case_total.C_dinc:1.4f}")
    print(f"Mean Thrust = {case_total.mean_thrust/1000.:1.4f} kN")
    print(f"Total Impulse = {case_total.total_impulse/1000.:1.4f} kN s")
    print(f"Mean Isp = {case_total.mean_Isp:1.4f} s")
    print(f"Consumed Oxidizer Mass = {case_total.m_ox_consumed:1.4f} kg")
    print(f"Consumed Fuel Mass = {case_total.m_fuel_consumed:1.4f} kg")
    print(f"Total Consumed Mass = {case_total.m_total_consumed:1.4f} kg")
    print(f"Mean Injector Pressure Ratio = {case_total.mean_injector_ratio:1.4f}\n\n")
    case_total.hybrid_burn_analysis(t_burn, 1., comp_correction=True)
    print("Compressible Analysis:")
    print(f"C_d = {case_total.C_dcomp:1.4f}")
    print(f"Mean Thrust = {case_total.mean_thrust/1000.:1.4f} kN")
    print(f"Total Impulse = {case_total.total_impulse/1000.:1.4f} kN s")
    print(f"Mean Isp = {case_total.mean_Isp:1.4f} s")
    print(f"Consumed Oxidizer Mass = {case_total.m_ox_consumed:1.4f} kg")
    print(f"Consumed Fuel Mass = {case_total.m_fuel_consumed:1.4f} kg")
    print(f"Total Consumed Mass = {case_total.m_total_consumed:1.4f} kg")
    print(f"Mean Injector Pressure Ratio = {case_total.mean_injector_ratio:1.4f}")
