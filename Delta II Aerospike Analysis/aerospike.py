#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 10:20:24 2021

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt
# importing the sys module
import sys
import scipy.optimize as optimize
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)

# inserting the mod.py directory at
# position 1 in sys.path
sys.path.insert(1, '/home/christian/Python Projects/mae-6530/MoC Nozzle Design')
sys.path.insert(1, '/home/christian/Python Projects/School Work/Fall 2019/MAE 5420')
sys.path.insert(1, '/home/christian/Python Projects/School Work/Fall 2018/Flight Sim/Flight-Simulator')


import moc
import KobayashiMaru
from stdatmos import statsi

comp_codes = KobayashiMaru.KobayashiMaru()

class Aerospike:
    def __init__(self, m_payload=1000.):
        self.m_payload = m_payload  # kg
        self.mu = 3.9860044e5  # Geocentric gravitational constant, km^3/s^2
        self.R_e = 6371  # km
        self.h_orbit = 860.06  # km
        self.R_e_ksc = 6373.18565  # km
        self.theta_orbit = np.deg2rad(28.7)  # rad
        self.p_sl = 101.325e3  # Pa
        self.m_shroud = 1040.  # kg
        self.R_g = 8314.4126  # J/K-mol
        self.g_0 = 9.8067  # m/s^2
        self.theta_1a = np.deg2rad(90.)  # rad
        self.theta_1b = np.deg2rad(30.)  # rad
        self.t_2a_ignition = 4.*60. + 37.7  # sec
        self.t_2a_jettison = 4.*60. + 56.  # sec
        self.t_2b_restart = 59.*60 + 21.  # sec

        # RS-27A Properties
        self.T_sl_rs27a = 890e3  # N
        self.T_vac_rs27a = 1085.8e3  # N
        self.exp_rat_rs27a = 15.2503
        self.theta_e_rs27a = np.deg2rad(30.5)  # rad
        self.P_0_rs27a = 5161.463e3  # Pa
        self.T_0_rs27a = 3455  # K
        self.gamma_rs27a = 1.222
        self.M_w_rs27a = 21.28  # kg/kg-mol
        self.m_p_rs27a = 97.08e3  # kg
        self.m_launch_rs27a = 101.8e3  # kg
        self.Rg_rs27a = self.R_g/self.M_w_rs27a  # J/K-mol

        # Gem-40 Properties
        self.T_sl_g40 = 442.95e3  # N
        self.T_vac_g40 = 499.20e3  # N
        self.exp_rat_g40 = 10.65
        self.theta_e_g40 = np.deg2rad(20.)  # rad
        self.P_0_g40 = 5652.66e3  # Pa
        self.T_0_g40 = 3600  # K
        self.gamma_g40 = 1.2
        self.M_w_g40 = 28.15  # kg/kg-mol
        self.m_p_g40 = 11.765e3  # kg
        self.m_launch_g40 = 13.080e3  # kg
        self.Rg_g40 = self.R_g/self.M_w_g40  # J/K-mol

        # AJ10-118 Aerojet Properties
        self.Isp_vac_aj = 319.  # s
        self.T_vac_aj = 43.657e3  # N
        self.T_sl_aj = 40.0e3  # N
        self.P_0_aj = 5700e3  # Pa
        self.T_0_aj = 3310  # K
        self.gamma_aj = 1.222
        self.M_w_aj = 21.7  # kg/kg-mol
        self.exp_rat_aj = 65.
        self.theta_e_aj = 0.  # rad
        self.m_p_aj = 6004.  # kg
        self.m_launch_aj = 6954.  # kg
        self.Rg_aj = self.R_g/self.M_w_aj  # J/K-mol

    def delta_v_req(self, verbose=True):
        V_orbit = np.sqrt(self.mu/(self.R_e + self.h_orbit))
        h_0 = 0.
        Omega_e = 2.*np.pi/((23 + 56/60 + 4.0905/3600)*3600)
        V_rot = (self.R_e_ksc + h_0)*Omega_e*np.cos(self.theta_orbit)
        DV_kinematic = V_orbit - V_rot
        DV_potential = np.sqrt(2.*self.mu*self.h_orbit/(self.R_e_ksc*(self.R_e_ksc + self.h_orbit)))
        self.DV_req = np.sqrt(DV_kinematic**2 + DV_potential**2)
        print(f"Kinematic Delta V: {DV_kinematic:.4f} km/s")
        print(f"Potential Delta V: {DV_potential:.4f} km/s")
        print(f"Required Delta V: {self.DV_req:.4f} km/s")

    def delta_v_available(self, aerospike=False, verbose=True):
        if not aerospike:
            self._exit_properties()
        m_init_1a, m_final_1a, Isp_eff_1a, t_burn_1a = self._stage_1a_analysis(aerospike, verbose)
        m_init_1b, m_final_1b, Isp_eff_1b, t_burn_1b = self._stage_1b_analysis(aerospike, verbose)
        m_init_2a, m_final_2a = self._stage_2a_analysis(verbose)
        m_init_2b, m_final_2b = self._stage_2b_analysis(verbose)
        m_frac_2b = (m_init_2b + self.m_payload)/(m_final_2b + self.m_payload)
        DV_2b = self.g_0*self.Isp_vac_aj*np.log(m_frac_2b)
        m_frac_2a = (m_init_2a + self.m_payload)/(m_final_2a + self.m_payload)
        DV_2a = self.g_0*self.Isp_vac_aj*np.log(m_frac_2a)
        m_frac_1b = (m_init_1b + m_init_2a + self.m_payload)/(m_final_1b + m_init_2a + self.m_payload)
        DV_1b_propulsive = 0.97*self.g_0*Isp_eff_1b*np.log(m_frac_1b)
        avg_grav_1b = ((2./3.)*(self.mu/(self.R_e_ksc + 16.31)**2) +
                       (1./3.)*(self.mu/(self.R_e_ksc + 105.52)**2))
        DV_1b_gravity = avg_grav_1b*t_burn_1b*np.sin(self.theta_1b)*1000
        DV_1b = DV_1b_propulsive - DV_1b_gravity
        m_frac_1a = (m_init_1a + m_init_2a + self.m_payload)/(m_final_1a + m_init_2a + self.m_payload)
        DV_1a_propulsive = 0.97*self.g_0*Isp_eff_1a*np.log(m_frac_1a)
        avg_grav_1a = ((2./3.)*(self.mu/(self.R_e_ksc + 0.)**2) +
                       (1./3.)*(self.mu/(self.R_e_ksc + 16.31)**2))
        DV_1a_gravity = avg_grav_1a*t_burn_1a*np.sin(self.theta_1a)*1000
        DV_1a = DV_1a_propulsive - DV_1a_gravity
        self.DV_available = (DV_1a + DV_1b + DV_2a + DV_2b)/1000.



    def _exit_properties(self):
        self.rs27a_exit_props = self._exit_conds(self.T_vac_rs27a, self.T_sl_rs27a,
                                                 self.exp_rat_rs27a, self.gamma_rs27a,
                                                 self.T_0_rs27a, self.P_0_rs27a,
                                                 self.Rg_rs27a, self.m_p_rs27a)
        self.g40_exit_props = self._exit_conds(self.T_vac_g40, self.T_sl_g40,
                                               self.exp_rat_g40, self.gamma_g40,
                                               self.T_0_g40, self.P_0_g40,
                                               self.Rg_g40, self.m_p_g40)
        self.aj_exit_props = self._exit_conds(self.T_vac_aj, self.T_sl_aj,
                                               self.exp_rat_aj, self.gamma_aj,
                                               self.T_0_aj, self.P_0_aj,
                                               self.Rg_aj, self.m_p_aj)

    def _exit_conds(self, F_vac, F_sl, exp_ratio, gamma, T_0, P_0, R_g, m_p):
        A_e = (F_vac - F_sl)/self.p_sl
        A_star = A_e/exp_ratio
        M_e = moc.A_Astar(A_e/A_star, gamma, 1.2)
        T_e = comp_codes.isenflowprops("T", M_e, gamma=gamma, stagnation=T_0)
        p_e = comp_codes.isenflowprops("P", M_e, gamma=gamma, stagnation=P_0)
        V_e = M_e*np.sqrt(gamma*R_g*T_e)
        mdot_choke = comp_codes.chokedmassflow(A_star, P_0, T_0, R_g, gamma=gamma)
        Isp_sl = F_sl/(self.g_0*mdot_choke)
        Isp_vac = F_vac/(self.g_0*mdot_choke)
        t_burn = m_p/mdot_choke
        prop_dict = {"exit": {"A_e": A_e,
                              "M_e": M_e,
                              "T_e": T_e,
                              "p_e": p_e,
                              "V_e": V_e},
                     "A_star": A_star,
                     "mdot": mdot_choke,
                     "Isp_sl": Isp_sl,
                     "Isp_vac": Isp_vac,
                     "t_burn": t_burn}
        return prop_dict

    def _stage1a_thrust_analysis(self, verbose):
        mdot_g40 = self.g40_exit_props["mdot"]
        V_e_g40 = self.g40_exit_props["exit"]["V_e"]
        p_e_g40 = self.g40_exit_props["exit"]["p_e"]
        A_e_g40 = self.g40_exit_props["exit"]["A_e"]
        theta_e_g40 = self.theta_e_g40

        mdot_rs27a = self.rs27a_exit_props["mdot"]
        V_e_rs27a = self.rs27a_exit_props["exit"]["V_e"]
        p_e_rs27a = self.rs27a_exit_props["exit"]["p_e"]
        A_e_rs27a = self.rs27a_exit_props["exit"]["A_e"]
        theta_e_rs27a = self.theta_e_rs27a

        alts = np.linspace(0., 16.31e3, 100)
        p_inf = np.zeros_like(alts)
        self.T_g40_1a = np.zeros_like(alts)
        self.T_rs27a_1a = np.zeros_like(alts)
        self.T_tot_1a = np.zeros_like(alts)
        for i in range(100):
            z, T, p_inf[i], d = statsi(alts[i])
            self.T_g40_1a[i] = self._thrust_eq(mdot_g40, V_e_g40,
                                               p_e_g40, p_inf[i], A_e_g40,
                                               theta_e_g40)
            self.T_rs27a_1a[i] = self._thrust_eq(mdot_rs27a, V_e_rs27a,
                                                 p_e_rs27a,
                                                 p_inf[i], A_e_rs27a,
                                                 theta_e_rs27a)
        self.T_tot_1a = 3.*self.T_g40_1a + self.T_rs27a_1a
        Isp_init_g40 = self._Isp(self.T_g40_1a[0], mdot_g40)
        Isp_final_g40 = self._Isp(self.T_g40_1a[-1], mdot_g40)
        Isp_init_rs27a = self._Isp(self.T_rs27a_1a[0], mdot_rs27a)
        Isp_final_rs27a = self._Isp(self.T_rs27a_1a[-1], mdot_rs27a)

        if verbose:
            plt.close('all')
            self.fig1, self.f_1ax1 = plt.subplots()
            self.f_1ax1.plot(alts/1000., self.T_rs27a_1a/1000., label='Baseline RS-27A',
                     color='k', linestyle=':')
            self.f_1ax1.plot(alts/1000., 3.*self.T_g40_1a/1000., label='3 x Gem 40',
                     color='k', linestyle='-.')
            self.f_1ax1.plot(alts/1000., self.T_tot_1a/1000., label='Total Thrust',
                     color='k', linestyle='-')
            self.f_1ax1.set_xlabel('Altitude, km')
            self.f_1ax1.set_ylabel('Thrust, kN')
            self.f_1ax1.set_yticks(np.arange(800, 2800, 200))
            self.f_1ax1.set_xlim((0, 18))
            self.f_1ax1.legend()
            print("RS-27A Thrust Parameters\n------------------------\n")
            print(f"Initial Thrust : {self.T_rs27a_1a[0]/1000.:.4f} kN")
            print(f"Final Thrust : {self.T_rs27a_1a[-1]/1000.:.4f} kN")
            print(f"Initial Isp : {Isp_init_rs27a:.4f} s")
            print(f"Final Isp : {Isp_final_rs27a:.4f} s\n")

            print("Gem 40 Thrust Parameters\n------------------------\n")
            print(f"Initial Thrust: {self.T_g40_1a[0]/1000.:.4f} kN")
            print(f"Final Thrust : {self.T_g40_1a[-1]/1000.:.4f} kN")
            print(f"Initial Isp : {Isp_init_g40:.4f} s")
            print(f"Final Isp : {Isp_final_g40:.4f} s\n")

    def _stage_1a_analysis(self, aerospike, verbose):
        """
        Lift Off to Gem 40 Jettison

        Returns
        -------
        None.

        """
        T_LO = self.T_sl_rs27a + self.T_sl_g40*3.

        if not aerospike:
            self._stage1a_thrust_analysis(verbose)

        t_burn_1a = self.g40_exit_props["t_burn"]
        mdot_rs27a = self.rs27a_exit_props["mdot"]
        mdot_g40 = self.g40_exit_props["mdot"]
        prop_burn_rs27a = mdot_rs27a*t_burn_1a
        prop_burn_g40 = mdot_g40*t_burn_1a
        prop_burn_1a = prop_burn_rs27a + 3.*prop_burn_g40
        m_liftoff_1a = self.m_launch_rs27a + 3.*self.m_launch_g40
        m_g40_burnout = m_liftoff_1a - prop_burn_1a
        m_g40_jettison = m_liftoff_1a - 3.*self.m_launch_g40 - prop_burn_rs27a
        m_rs27a_1a = self.m_p_rs27a - prop_burn_rs27a
        self.m_init_1b = m_g40_jettison
        self.m_prop_init_1b = m_rs27a_1a

        Isp_launch = self._Isp(self.T_sl_rs27a + 3.*self.T_sl_g40,
                               mdot_rs27a + 3.*mdot_g40)
        T_g40_burnout = self.T_rs27a_1a[-1] + 3.*self.T_g40_1a[-1]
        Isp_g40_burnout = self._Isp(T_g40_burnout, mdot_rs27a + 3.*mdot_g40)
        Isp_eff_1a = (2./3.)*Isp_launch + (1./3.)*Isp_g40_burnout
        if verbose:
            print("---------------------\n| Stage 1a Analysis |\n---------------------\n")
            print(f"Total Lift Off Thrust : {T_LO/1000.:4f} kN\n")
            print(f"Gem 40 Burn Time : {self.g40_exit_props['t_burn']:.4f} s\n")
            print("Masses Through Stage 1a\n----------------------\n")
            print(f"Lift Off Mass : {m_liftoff_1a:4g} kg")
            print(f"Mass After Gem 40 Burnout : {m_g40_burnout:.4f} kg")
            print(f"Mass After Gem 40 Jettison : {m_g40_jettison:.4f} kg")
            print(f"Remaining RS-27A Propellant Mass : {m_rs27a_1a:.4f} kg\n")
            print(f"Effective Specific Impulse : {Isp_eff_1a:.4f} s\n")
        return m_liftoff_1a, m_g40_burnout, Isp_eff_1a, t_burn_1a

    def _stage_1b_analysis(self, aerospike, verbose):
        """
        Gem 40 Jettison to MECO (Main Engine Cutoff)

        Returns
        -------
        None.

        """
        t_burn_1b = self.rs27a_exit_props["t_burn"] - self.g40_exit_props["t_burn"]

        mdot_rs27a = self.rs27a_exit_props["mdot"]
        V_e_rs27a = self.rs27a_exit_props["exit"]["V_e"]
        p_e_rs27a = self.rs27a_exit_props["exit"]["p_e"]
        A_e_rs27a = self.rs27a_exit_props["exit"]["A_e"]
        theta_e_rs27a = self.theta_e_rs27a

        alts = np.linspace(16.31e3, 105.52e3, 100)
        if not aerospike:
            p_inf = np.zeros_like(alts)
            self.T_rs27a_1b = np.zeros_like(alts)
            for i in range(100):
                z, T, p_inf[i], d = statsi(alts[i])
                self.T_rs27a_1b[i] = self._thrust_eq(mdot_rs27a, V_e_rs27a,
                                                     p_e_rs27a,
                                                     p_inf[i], A_e_rs27a,
                                                     theta_e_rs27a)
        Isp_init_1b = self._Isp(self.T_rs27a_1b[0], mdot_rs27a)
        Isp_final_1b = self._Isp(self.T_rs27a_1b[-1], mdot_rs27a)


        prop_burn_1b = mdot_rs27a*t_burn_1b
        m_MECO = self.m_init_1b - prop_burn_1b
        m_final_rs27a = self.m_prop_init_1b - prop_burn_1b

        Isp_eff_1b = (2./3.)*Isp_init_1b + (1./3.)*Isp_final_1b
        if verbose:
            print("---------------------\n| Stage 1b Analysis |\n---------------------\n")
            print(f"Burn Time from Gem 40 Burnout to MECO : {t_burn_1b:.4f} s\n")
            if not aerospike:
                self.fig2, self.f_2ax1 = plt.subplots()
                self.f_2ax1.plot(alts/1000., self.T_rs27a_1b/1000., color='k', label='Baseline RS-27A')
                self.f_2ax1.set_xlabel('Altitude, km')
                self.f_2ax1.set_ylabel('Thrust, kN')
                # self.f_2ax1.set_yticks(np.arange(1066, 1088, 2))
                self.f_2ax1.set_xlim((0, 120))
            print("RS-27A Thrust Parameters\n------------------------\n")
            print(f"Initial Thrust : {self.T_rs27a_1b[0]/1000.:.4f} kN")
            print(f"Final Thrust : {self.T_rs27a_1b[-1]/1000.:.4f} kN")
            print(f"Initial Isp : {Isp_init_1b:.4f} s")
            print(f"Final Isp : {Isp_final_1b:.4f} s\n")
            print("Masses Through Stage 1b\n----------------------\n")
            print(f"Initial Total Mass : {self.m_init_1b:.4f} kg")
            print(f"Initial Propellant Mass : {self.m_prop_init_1b:.4f} kg")
            print(f"Mass at MECO : {m_MECO:.4f} kg")
            print(f"Remaining RS-27A Propellant Mass : {np.round(abs(m_final_rs27a), 2):.1f} kg\n")
            print(f"Effective Specific Impulse : {Isp_eff_1b:.4f} s\n")
        return self.m_init_1b, m_MECO, Isp_eff_1b, t_burn_1b

    def _stage_2a_analysis(self, verbose):
        """
        Stage 2 Ignition (Aerojet Engine) to Fairing Jettison

        Returns
        -------
        None.

        """
        mdot_aj = self.T_vac_aj/(self.g_0*self.Isp_vac_aj)

        t_burn_2a = self.t_2a_jettison - self.t_2a_ignition

        prop_burn_2a = mdot_aj*t_burn_2a
        m_init_2a = self.m_launch_aj + self.m_shroud
        m_before_jettison = m_init_2a - prop_burn_2a
        m_after_jettison = m_before_jettison - self.m_shroud
        m_final_aj = self.m_p_aj - prop_burn_2a
        self.m_prop_2b = m_final_aj
        self.m_init_2b = m_after_jettison
        if verbose:
            print("---------------------\n| Stage 2a Analysis |\n---------------------\n")
            print(f"AJ10-118K Aerojet Engine Massflow : {mdot_aj:.4f} kg/s\n")
            print(f"Burn Time from Stage 2 Ignition to Fairing Jettison : {t_burn_2a:.4f} s\n")
            print("AJ10-118K Thrust Parameters\n------------------------\n")
            print(f"Initial Thrust : {self.T_vac_aj/1000.:.4f} kN")
            print(f"Initial Isp : {self.Isp_vac_aj:.4f} s\n")
            print("Masses Through Stage 2a\n----------------------\n")
            print(f"Initial Total Mass : {m_init_2a:.4f} kg")
            print(f"Initial Propellant Mass : {self.m_p_aj:.4f} kg")
            print(f"Mass Before Shroud Jettison : {m_before_jettison:.4f} kg")
            print(f"Mass After Shroud Jettison : {m_after_jettison:.4f} kg")
            print(f"Remaining Stage 2 Propellant Mass : {m_final_aj:.4f} kg\n")
        return m_init_2a, m_before_jettison



    def _stage_2b_analysis(self, verbose):
        """
        Fairing Jettison to Second-Stage Restart

        Returns
        -------
        None.

        """
        prop_burn_2b = self.m_prop_2b
        m_init_2b = self.m_init_2b
        m_final_2b = self.m_launch_aj - self.m_p_aj
        if verbose:
            print("---------------------\n| Stage 2b Analysis |\n---------------------\n")
            print("Masses Through Stage 2b\n----------------------\n")
            print(f"Initial Total Mass (excluding payload) : {m_init_2b:.4f} kg")
            print(f"Propellant Consumed : {prop_burn_2b:.4f} kg")
            print(f"Final Total Mass (excluding payload) : {m_final_2b:.4f} kg")
        return m_init_2b, m_final_2b

    def _Isp(self, F, mdot):
        return F/(self.g_0*mdot)

    def _thrust_eq(self, mdot, V_e, p_e, p_inf, A_e, theta_e):
        gamma_e = 0.5*(1 + np.cos(theta_e))
        C1 = mdot*V_e*gamma_e
        C2 = 0.
        C3 = (p_e - p_inf)*A_e
        return C1 - C2 + C3

    def aerospike_replacement(self):
        M_e = self.rs27a_exit_props["exit"]["M_e"]
        A_e = self.rs27a_exit_props["exit"]["A_e"]
        R_e = np.sqrt(A_e/np.pi)
        self.nu_e_rs27a = moc._nu_M(M_e, self.gamma_rs27a)
        M_P = np.linspace(1.0, M_e, 300)
        nu_P = np.array([moc._nu_M(m, self.gamma_rs27a) for m in M_P])
        mu_P = np.array([np.arcsin(1./m) for m in M_P])
        Y_P = np.array([self._aerospike_radius(R_e, self.nu_e_rs27a, n, m,
                                               self.exp_rat_rs27a, M,
                                               self.gamma_rs27a) for n, m, M in zip(nu_P, mu_P, M_P)])
        X_P = np.array([self._aerospike_axial(R_e, y, self.nu_e_rs27a, n, m) for y, n, m in zip(Y_P, nu_P, mu_P)])
        self.f_3ax1.plot((self.L_N + X_P)*100., Y_P*100., color='r')
        self.f_3ax1.plot((self.L_N + X_P)*100., -Y_P*100., color='r', label='Aerospike')

        P_0 = self.P_0_rs27a
        gamma = self.gamma_rs27a
        h_design = optimize.minimize(self._spike_design_alt, 0.,
                                     args=(P_0, gamma, self.nu_e_rs27a)).x[0]
        print("---------------------\n| Aerospike Analysis |\n---------------------\n")
        print(f"Design Altitude: {h_design/1000.:.4f} km")
        p_P = np.array([comp_codes.isenflowprops("P", m, stagnation=P_0) for m in M_P])
        self.fig6, self.f_6ax1 = plt.subplots()

        self.f_6ax2 = self.f_6ax1.twinx()
        self.f_6ax1.plot(X_P*100., p_P/1000., 'r-')
        self.f_6ax2.plot(X_P*100., M_P, 'b-')
        self.f_6ax1.set_yscale("log")
        self.f_6ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        self.f_6ax1.set_yticks([10, 100, 1000, 10000])
        self.f_6ax2.set_yticks(np.arange(1, 4.1, 0.5))
        self.f_6ax2.set_ylim((1, 4))
        self.f_6ax1.set_xlim((X_P[0]*100., 300))

        self.f_6ax1.set_xlabel('$X$ Along Spike, cm')
        self.f_6ax1.set_ylabel('$p$, kPa', color='r')
        self.f_6ax2.set_ylabel('$M$', color='b')

        trunc_mask = p_P >= self.p_sl
        self.f_3ax1.plot((self.L_N + X_P[trunc_mask])*100., Y_P[trunc_mask]*100., 'b-.')
        self.f_3ax1.plot((self.L_N + X_P[trunc_mask])*100., -Y_P[trunc_mask]*100., 'b-.', label='Truncated Aerospike')
        self.f_3ax1.vlines((self.L_N + X_P[trunc_mask][-1])*100., ymin=-Y_P[trunc_mask][-1]*100., ymax=Y_P[trunc_mask][-1]*100., color='b', linestyle='-.')
        self.f_3ax1.legend(loc='upper right')

        self.f_6ax1.scatter([X_P[trunc_mask][-1]*100.], [p_P[trunc_mask][-1]/1000.], facecolor='r')
        self.f_6ax2.scatter([X_P[trunc_mask][-1]*100.], [M_P[trunc_mask][-1]], facecolor='b')
        self.X_P = X_P
        self.Y_P = Y_P
        self._aerospike_thrust_calc(p_P, Y_P, self.nu_e_rs27a, trunc_mask)
        self.T_sl_rs27a = self.T_rs27a_trunc_1a[0]
        self.T_vac_rs27a = self.T_rs27a_trunc_1b[-1]
        self.T_rs27a_1a = self.T_rs27a_trunc_1a
        self.T_rs27a_1b = self.T_rs27a_trunc_1b


    def _aerospike_radius(self, R_e, nu_e, nu_x, mu_x, eps, M_x, gamma):
        C1 = (np.sin(nu_e - nu_x + mu_x)/eps)
        C2 = (2./(gamma + 1.))*(1. + (gamma - 1.)*M_x*M_x/2.)
        E1 = (gamma + 1)/(2*(gamma - 1))
        if 1. - C1*(C2**E1) < 0:
            R_x = 0.
        else:
            R_x = R_e*np.sqrt(1. - C1*(C2**E1))
        return R_x

    def _aerospike_axial(self, R_e, R_x, nu_e, nu_x, mu_x):
        num = R_e - R_x
        denom = np.tan(nu_e - nu_x + mu_x)
        return num/denom

    def _spike_design_alt(self, h, P_0, gamma, nu_e):
        z, T, p_amb, d = statsi(h)
        M_exp = comp_codes.isenflowprops("MP", 0., gamma=gamma,
                                         stagnation=P_0, flow=p_amb)
        nu_exp = moc._nu_M(M_exp, gamma)
        return abs(nu_exp - nu_e)

    def _aerospike_expansion_line(self, h):
        P_0 = self.P_0_rs27a
        gamma = self.gamma_rs27a
        nu_e = self.nu_e_rs27a
        z, T, p_amb, d = statsi(h)
        M_exp = comp_codes.isenflowprops("MP", 0., gamma=gamma,
                                         stagnation=P_0, flow=p_amb)
        nu_exp = moc._nu_M(M_exp, gamma)
        theta_exp = nu_exp - nu_e
        h, l = self.f_3ax1.get_legend_handles_labels()
        if "Expansion Line" in l:
            self.f_3ax1.lines[-1].remove()
            self.f_3ax1.get_lines()
        startx = self.X_P[0] + self.L_N
        starty = self.Y_P[0]
        endx = self.X_P[-1] + self.L_N
        endy = starty + np.tan(theta_exp)*endx
        self.f_3ax1.plot([startx*100., endx*100.], [starty*100., endy*100.], 'g:', label='Expansion Line')
        self.f_3ax1.legend()

    def _aerospike_thrust_calc(self, p_P, Y_P, nu_e, trunc_mask):
        delta_throat = np.pi/2. - nu_e
        mdot = self.rs27a_exit_props["mdot"]
        gamma = self.gamma_rs27a
        R_g = self.Rg_rs27a
        A_star = self.rs27a_exit_props["A_star"]
        T_0 = self.T_0_rs27a
        P_0 = self.P_0_rs27a
        T_star = T_0/(0.5*(gamma + 1.))
        p_star = P_0/((0.5*(gamma + 1.))**(gamma/(gamma - 1.)))
        alts = np.linspace(0., 16.31e3, 100)
        p_inf = np.zeros_like(alts)
        self.F_throat_as_1a = np.zeros_like(alts)
        self.F_ramp_as_1a = np.zeros_like(alts)
        self.F_ramp_trunc_1a = np.zeros_like(alts)
        self.F_base_1a = np.zeros_like(alts)
        for i in range(len(alts)):
            z, T, p_inf[i], d = statsi(alts[i])
            self.F_throat_as_1a[i] = (mdot*np.sqrt(gamma*R_g*T_star) + (p_star - p_inf[i])*A_star)*np.sin(delta_throat)
            for j in range(len(p_P) - 1):
                self.F_ramp_as_1a[i] += (0.5*(p_P[j] + p_P[j+1]) - p_inf[i])*np.pi*(Y_P[j]**2 - Y_P[j+1]**2)
                if p_P[j] > p_P[trunc_mask][-1]:
                    self.F_ramp_trunc_1a[i] = self.F_ramp_as_1a[i]
            self.F_base_1a[i] = self._aerospike_base_thrust(self.F_ramp_as_1a[i], self.F_ramp_trunc_1a[i])
        self.f_1ax1.plot(alts/1000., (self.F_throat_as_1a + self.F_ramp_as_1a)/1000., 'r:', label='RS-27A w/ Full Aerospike')
        self.f_1ax1.plot(alts/1000., (self.F_throat_as_1a + self.F_ramp_as_1a)/1000. + 3.*self.T_g40_1a/1000., 'r-', label='Total Thrust w/ Full Aerospike')
        self.f_1ax1.plot(alts/1000., (self.F_throat_as_1a + self.F_ramp_trunc_1a + self.F_base_1a)/1000., 'g--', label='RS-27A w/ Truncated Aerospike')
        self.f_1ax1.plot(alts/1000., (self.F_throat_as_1a + self.F_ramp_trunc_1a + self.F_base_1a + 3.*self.T_g40_1a)/1000., 'g-', label='Total Thrust w/ Truncated Aerospike')
        self.f_1ax1.legend()

        mdot_1a = mdot + 3.*self.g40_exit_props["mdot"]
        thrust_as_1a = self.F_throat_as_1a + self.F_ramp_as_1a + 3.*self.T_g40_1a
        thrust_trunc_1a = self.F_throat_as_1a + self.F_ramp_trunc_1a + self.F_base_1a + 3.*self.T_g40_1a
        self.Isp_1a_as = np.array([self._Isp(f, mdot_1a) for f in thrust_as_1a])
        self.Isp_1a_trunc = np.array([self._Isp(f, mdot_1a) for f in thrust_trunc_1a])
        self.f_4ax1.plot(alts/1000., self.Isp_1a_as, 'r-', label='RS-27A w/ Full Aerospike')
        self.f_4ax1.plot(alts/1000., self.Isp_1a_trunc, 'g-', label='RS-27A w/ Truncated Aerospike')
        self.f_4ax1.legend()
        self.T_rs27a_trunc_1a = self.F_throat_as_1a + self.F_ramp_trunc_1a + self.F_base_1a


        alts = np.linspace(16.31e3, 105.52e3, 100)
        p_inf = np.zeros_like(alts)
        self.F_throat_as_1b = np.zeros_like(alts)
        self.F_ramp_as_1b = np.zeros_like(alts)
        self.F_ramp_trunc_1b = np.zeros_like(alts)
        self.F_base_1b = np.zeros_like(alts)
        for i in range(len(alts)):
            z, T, p_inf[i], d = statsi(alts[i])
            self.F_throat_as_1b[i] = (mdot*np.sqrt(gamma*R_g*T_star) + (p_star - p_inf[i])*A_star)*np.sin(delta_throat)
            for j in range(len(p_P) - 1):
                self.F_ramp_as_1b[i] += (0.5*(p_P[j] + p_P[j+1]) - p_inf[i])*np.pi*(Y_P[j]**2 - Y_P[j+1]**2)
                if p_P[j] > p_P[trunc_mask][-1]:
                    self.F_ramp_trunc_1b[i] = self.F_ramp_as_1b[i]
            self.F_base_1b[i] = self._aerospike_base_thrust(self.F_ramp_as_1b[i], self.F_ramp_trunc_1b[i])
        self.f_2ax1.plot(alts/1000., (self.F_throat_as_1b + self.F_ramp_as_1b)/1000., 'r-', label='RS-27A w/ Full Aerospike')
        self.f_2ax1.plot(alts/1000., (self.F_throat_as_1b + self.F_ramp_trunc_1b + self.F_base_1b)/1000., 'g-', label='RS-27A w/ Truncated Aerospike')
        self.f_2ax1.legend()

        thrust_as_1b = self.F_throat_as_1b + self.F_ramp_as_1b
        thrust_trunc_1b = self.F_throat_as_1b + self.F_ramp_trunc_1b + self.F_base_1b
        self.Isp_1b_as = np.array([self._Isp(f, mdot) for f in thrust_as_1b])
        self.Isp_1b_trunc = np.array([self._Isp(f, mdot) for f in thrust_trunc_1b])
        self.f_5ax1.plot(alts/1000., self.Isp_1b_as, 'r-', label='RS-27A w/ Full Aerospike')
        self.f_5ax1.plot(alts/1000., self.Isp_1b_trunc, 'g-', label='RS-27A w/ Truncated Aerospike')
        self.f_5ax1.legend()
        self.T_rs27a_trunc_1b = thrust_trunc_1b


    def min_length_rs27a(self):
        M_e = self.rs27a_exit_props["exit"]["M_e"]
        A_t = self.rs27a_exit_props["A_star"]
        exp = self.exp_rat_rs27a
        gamma = self.gamma_rs27a
        D_t = 2.*np.sqrt(A_t/np.pi)
        R_c = 0.
        theta_e = self.theta_e_rs27a
        self.L_N = self._nozzle_length(exp, D_t, R_c, theta_e)
        x_s = np.linspace(0., self.L_N)
        y_s = 0.5*D_t + np.tan(theta_e)*x_s
        baseline_config = {"color": 'k', "linestyle": '-'}
        min_config = {"color": 'k', "linestyle": '-.'}
        self.fig3, self.f_3ax1 = plt.subplots()
        self.f_3ax1.plot(x_s*100., y_s*100., **baseline_config)
        self.f_3ax1.plot(x_s*100., -y_s*100., **baseline_config)
        self.f_3ax1.plot([self.L_N*100., self.L_N*100.], [y_s[-1]*100., -y_s[-1]*100.], **baseline_config, label="RS-27A")

        theta_e = moc._nu_M(M_e, gamma)/2.
        L_N = self._nozzle_length(exp, D_t, R_c, theta_e)
        x_s = np.linspace(0., L_N)
        y_s = 0.5*D_t + np.tan(theta_e)*x_s
        self.f_3ax1.plot(x_s*100., y_s*100., **min_config)
        self.f_3ax1.plot(x_s*100., -y_s*100., **min_config)
        self.f_3ax1.plot([L_N*100., L_N*100.], [y_s[-1]*100., -y_s[-1]*100.], **min_config, label="Min. Length RS-27A")
        mdot = self.rs27a_exit_props["mdot"]
        V_e = self.rs27a_exit_props["exit"]["V_e"]
        p_e = self.rs27a_exit_props["exit"]["p_e"]
        A_e = self.rs27a_exit_props["exit"]["A_e"]
        alts = np.linspace(0., 16.31e3, 100)
        p_inf = np.zeros_like(alts)
        self.T_rs27a_min_1a = np.zeros_like(alts)
        self.Isp_rs27a_min_1a = np.zeros_like(alts)
        for i in range(len(alts)):
            z, T, p_inf[i], d = statsi(alts[i])
            self.T_rs27a_min_1a[i] = self._thrust_eq(mdot, V_e, p_e, p_inf[i], A_e, theta_e)
            self.Isp_rs27a_min_1a[i] = self._Isp(self.T_rs27a_min_1a[i] + 3.*self.T_g40_1a[i], mdot + 3.*self.g40_exit_props["mdot"])
        self.f_1ax1.plot(alts/1000., (self.T_rs27a_min_1a)/1000., 'b:', label='RS-27A Min. Length')
        self.fig4, self.f_4ax1 = plt.subplots()
        self.f_4ax1.plot(alts/1000., self.Isp_rs27a_min_1a, 'k-', label='RS-27A Min. Length')
        self.f_4ax1.set_ylabel('Isp, s')
        self.f_4ax1.set_xlabel('Altitude, km')
        self.f_4ax1.set_xlim(0, 18)
        self.f_4ax1.set_xticks(np.arange(0, 19, 2))

        alts = np.linspace(16.31e3, 105.52e3, 100)
        p_inf = np.zeros_like(alts)
        self.T_rs27a_min_1b = np.zeros_like(alts)
        self.Isp_rs27a_min_1b = np.zeros_like(alts)
        for i in range(len(alts)):
            z, T, p_inf[i], d = statsi(alts[i])
            self.T_rs27a_min_1b[i] = self._thrust_eq(mdot, V_e, p_e, p_inf[i], A_e, theta_e)
            self.Isp_rs27a_min_1b[i] = self._Isp(self.T_rs27a_min_1b[i], mdot)
        self.fig5, self.f_5ax1 = plt.subplots()
        self.f_5ax1.plot(alts/1000., self.Isp_rs27a_min_1b, 'k-', label='RS-27A Min. Length')
        self.f_5ax1.set_ylabel('Isp, s')
        self.f_5ax1.set_xlabel('Altitude, km')
        self.f_5ax1.set_xlim(0, 120)



    def _aerospike_base_thrust(self, F_max, F_accum):
        return 0.58*(F_max - F_accum)

    def _nozzle_length(self, exp, D_t, R_c, theta_e):
        C1 = 0.5*D_t*(np.sqrt(exp) - 1.)
        C2 = R_c*(1./np.cos(theta_e) - 1.)
        L_N = (C1 + C2)/np.tan(theta_e)
        return L_N




def find_max_payload(m_payload, case, verbose, aerospike):
    case.m_payload = m_payload
    case.delta_v_available(aerospike, verbose)
    return abs(case.DV_req - case.DV_available)

if __name__ == "__main__":
    case1 = Aerospike()
    print("\n-----------------------------\n| Delta V Required Analysis |\n-----------------------------\n")
    case1.delta_v_req()
    print("\n------------------------------\n| Delta V Available Analysis |\n------------------------------\n")
    case1.delta_v_available()

    opt_payload = optimize.minimize(find_max_payload, 1000., args=(case1, False, False)).x[0]
    print("\n----------------------------\n| Delta II Maximum Payload |\n----------------------------\n")
    print(f"Payload : {opt_payload:.4f} kg\n")

    case1.min_length_rs27a()
    case1.aerospike_replacement()
    print("\n----------------------------------------------\n| Delta II Analysis With Truncated Aerospike |\n----------------------------------------------\n")
    case1.delta_v_available(aerospike=True)
    opt_payload_as = optimize.minimize(find_max_payload, 1000., args=(case1, False, True)).x[0]
    print("\n-------------------------------------------\n| Delta II Maximum Payload With Aerospike |\n-------------------------------------------\n")
    print(f"Payload : {opt_payload_as:.4f} kg\n")
#

