#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 15:01:17 2021

@author: christian
"""
import numpy as np
import matplotlib.pyplot as plt
# importing the sys module
import sys
import scipy.optimize as optimize

# inserting the mod.py directory at
# position 1 in sys.path
sys.path.insert(1, '/home/christian/Python Projects/mae-6530/MoC Nozzle Design')
sys.path.insert(1, '/home/christian/Python Projects/School Work/Fall 2019/MAE 5420')
sys.path.insert(1, '/home/christian/Python Projects/School Work/Fall 2018/Flight Sim/Flight-Simulator')


import moc
import KobayashiMaru
from stdatmos import statsi
import DiamondWedge
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'

comp_codes = KobayashiMaru.KobayashiMaru()
oblique_shock = DiamondWedge.DiamondWedge(0., 1.0)

class ConicalShock:
    def __init__(self, M_inf, beta_init, N=100):
        self.beta_init = beta_init
        self.M_inf = M_inf
        self.gamma = 1.4
        self.N = N
        comp_codes.gamma = self.gamma

    def _properties_behind_shock(self, beta):
        Mn1 = self.M_inf*np.sin(beta)
        Mn2 = oblique_shock.obliqueshockmach(Mn1)
        delta = oblique_shock.Th_Be_Ma(beta, self.M_inf)
        M2 = Mn2/(np.sin(beta - delta))
        p_ratio, T_ratio = oblique_shock.obliqueshock(self.M_inf, beta, delta)[1:3]
        return M2, delta, p_ratio, T_ratio

    def _script_V(self, M2):
        N = 0.5*(self.gamma - 1)*M2*M2
        D = 1. + N
        V = np.sqrt(N/D)
        return V

    def _M_f_script_V(self, V_cone):
        C1 = 2./(self.gamma - 1.)
        C2 = (V_cone**2)/(1. - V_cone**2)
        M_cone = np.sqrt(C1*C2)
        return M_cone

    def _F_script_V(self, V, theta):
        V_r, V_t = V
        N1 = V_t**2*V_r
        N2 = 0.5*(self.gamma - 1.)*(1. - V_r**2 - V_t**2)
        N3 = 2.*V_r + V_t/np.tan(theta)
        N = N1 - N2*N3
        D = N2 - V_t**2
        return np.array([V_t, N/D])

    def _P0_drop_normal(self, M1, M2):
        C1 = M1/M2
        N2 = 2. + (self.gamma - 1.)*M2**2
        D2 = 2. + (self.gamma - 1.)*M1**2
        C2 = N2/D2
        E1 = (self.gamma + 1.)/(2.*(self.gamma - 1.))
        P0_drop_ratio = C1*C2**E1
        return P0_drop_ratio

    def _isen_flow_ratios(self, flag, M):
        C = (1. + 0.5*(self.gamma - 1.)*M*M)
        if flag == "P":
            exp = self.gamma/(self.gamma - 1.)
            return C**exp
        if flag == "rho":
            exp = 1./(self.gamma - 1.)
            return C**exp
        if flag == "T":
            return C

    def _iteration_to_cone(self, beta):
        self.M_1, delta, p_ratio, T_ratio = self._properties_behind_shock(beta)
        V_mag = self._script_V(self.M_1)
        V_r = V_mag*np.cos(beta - delta)
        V_t = -V_mag*np.sin(beta - delta)
        V_i = np.array([V_r, V_t])
        V_final = [V_i]
        d_theta = beta/self.N
        theta_i = beta
        while V_i[1] < 0.:
            F_V_predictor = self._F_script_V(V_i, theta_i)
            V_predictor = V_i - d_theta*F_V_predictor
            F_V_corrector = self._F_script_V(V_predictor, theta_i - d_theta)
            V_corrector = V_i - 0.5*d_theta*(F_V_predictor + F_V_corrector)
            V_i = V_corrector
            theta_i -= d_theta
            V_final.append(V_i)
        return theta_i, V_i, V_final

    def taylor_maccoll(self, cone_angle, opt=False):
        self.beta = self.beta_init
        theta_i = 100.
        num_iterations = 0
        d_theta = 0.
        while (abs(np.rad2deg(theta_i - cone_angle)) > 1e-5)*(num_iterations < 500):
            self.beta -= (2./3.)*d_theta
            num_iterations += 1
            theta_i, V_i, V_final = self._iteration_to_cone(self.beta)
            d_theta = theta_i - cone_angle
        if not opt:
            print(f"Final Propagated Cone Angle: {theta_i*180/np.pi:.4f} deg")
            print(f"Conical Shock Angle: {self.beta*180/np.pi:.4f} deg")
            y_cowl = 40.  # cm
            x_cowl = y_cowl/np.tan(self.beta)
            print(f"Longitudinal Distance from Spike Tip to Cowl Inlet: {x_cowl:.4f} cm")
            R_cowl = np.sqrt(x_cowl**2 + y_cowl**2)
            print(f"Radial Distance from Spike Tip to Cowl Inlet: {R_cowl:.4f} cm")

        theta = np.linspace(self.beta, cone_angle, len(V_final))
        Vmag_final = [np.sqrt(v[0]**2 + v[1]**2) for v in V_final]
        M_dist_1 = [self._M_f_script_V(v) for v in Vmag_final]

        Mn1 = self.M_inf*np.sin(self.beta)
        P01_P0inf = comp_codes.stagpressnormshock(Mn1)
        P0inf_pinf = self._isen_flow_ratios("P", self.M_inf)
        p1_P01 = [1./self._isen_flow_ratios("P", m) for m in M_dist_1]
        P0_dist_1 = np.full(len(M_dist_1), P01_P0inf)
        p1_pinf_dist = [P0inf_pinf*P01_P0inf*p for p in p1_P01]

        T01_T0inf = 1.
        T0inf_Tinf = self._isen_flow_ratios("T", self.M_inf)
        T1_T01 = [1./ self._isen_flow_ratios("T", m) for m in M_dist_1]
        T1_Tinf_dist = [T0inf_Tinf*T01_T0inf*t for t in T1_T01]

        if not opt:
            fig1, ax1 = plt.subplots()
            ax1.axhline(self.M_inf, color='k', linestyle='-.', label=r'$M_\infty$')
            ax1.plot(np.rad2deg(theta), M_dist_1, color='k', label=r'$M_\theta$')
            ax1.set_xlim(np.ceil(np.rad2deg(self.beta)), np.rad2deg(cone_angle))
            ax1.set_ylabel('Mach Number', fontsize=16)
            ax1.set_xlabel(r'$\theta_\mathrm{inlet}$, deg', fontsize=16)
            ax1.tick_params(axis='x', labelsize=12)
            ax1.tick_params(axis='y', labelsize=12)
            ax1.legend()
            fig2, ax2 = plt.subplots()
            ax2_2 = ax2.twinx()
            ax2.plot(np.rad2deg(theta), p1_pinf_dist, label=r'$\frac{p_1}{p_\infty}$', color='r', linestyle='-')
            ax2.plot(np.rad2deg(theta), P0_dist_1, label=r'$\frac{P_{0_1}}{P_{0_\infty}}$', color='r', linestyle='-.')
            ax2_2.plot(np.rad2deg(theta), T1_Tinf_dist, label=r'$\frac{T_1}{T_\infty}$', color='b', linestyle='-')
            ax2.set_xlim(np.ceil(np.rad2deg(self.beta)), np.rad2deg(cone_angle))
            ax2.set_ylim(0.97, 1.47)
            ax2_2.set_ylim(1.045, 1.115)
            ax2.set_ylabel(r'Pressure Ratio', color='r', fontsize=16)
            ax2_2.set_ylabel(r'Temperature Ratio', color='b', fontsize=16)
            ax2.set_xlabel(r'$\theta_\mathrm{inlet}$, deg', fontsize=16)
            ax2.tick_params(axis='x', labelsize=12)
            ax2.tick_params(axis='y', labelsize=12)
            ax2_2.tick_params(axis='y', labelsize=12)
            ax2.legend(prop={'size': 14})
            ax2_2.legend(loc='upper center', prop={'size': 14})

        Mavg_1 = np.trapz(np.flip(M_dist_1*np.sin(theta)), x=np.flip(theta))/(np.cos(cone_angle) - np.cos(self.beta))
        pavg_1 = np.trapz(np.flip(p1_pinf_dist*np.sin(theta)), x=np.flip(theta))/(np.cos(cone_angle) - np.cos(self.beta))
        if not opt:
            print(f"Area-Weighted Mean Mach Number at Cowl Inlet: {Mavg_1:.4f}")
            print(f"Area-Weighted Mean Pressure Ratio at Cowl Inlet: {pavg_1:.4f}")

        self.M_2 = comp_codes.normalshockmach(Mavg_1)
        P02_P01 = comp_codes.stagpressnormshock(Mavg_1)
        P02_P0inf = P01_P0inf*P02_P01
        P01_p1 = self._isen_flow_ratios("P", Mavg_1)
        p2_P02 = 1./self._isen_flow_ratios("P", self.M_2)
        p2_p1 = P01_p1*P02_P01*p2_P02
        if not opt:
            print(f"Mach Number Behind Cowl Shock: {self.M_2:.4f}")
            print(f"Compression Ratio Through Cowl Shock: {p2_p1:.4f}")
            print(f"Ratio of Stagnation Pressure Behind Cowl Shock to Freestream: {P02_P0inf:.4f}")
        return P02_P0inf

    def twoD_solution(self, cone_angle, opt=False):
        beta = oblique_shock.Be_Th_Maexplicit(self.M_inf, cone_angle)
        if not opt:
            print(f"Cone Angle: {cone_angle*180/np.pi:.4f} deg")
            print(f"Conical Shock Angle: {beta*180./np.pi:.4f} deg")
            y_cowl = 40.  # cm
            x_cowl = y_cowl/np.tan(beta)
            print(f"Longitudinal Distance from Spike Tip to Cowl Inlet: {x_cowl:.4f} cm")
            R_cowl = np.sqrt(x_cowl**2 + y_cowl**2)
            print(f"Radial Distance from Spike Tip to Cowl Inlet: {R_cowl:.4f} cm")
        M_1, p1_pinf, T1_Tinf, rho1_rhoinf = oblique_shock.obliqueshock(self.M_inf, beta, cone_angle)
        M_ninf = self.M_inf*np.sin(beta)
        P01_P0inf = comp_codes.stagpressnormshock(M_ninf)
        P0inf_pinf = self._isen_flow_ratios("P", self.M_inf)
        p1_P01 = 1./self._isen_flow_ratios("P", M_1)
        p1_pinf = P0inf_pinf*P01_P0inf*p1_P01
        if not opt:
            print(f"Mach Number at Cowl Inlet: {M_1:.4f}")
            print(f"Pressure Ratio at Cowl Inlet: {p1_pinf:.4f}")
        M_2 = comp_codes.normalshockmach(M_1)
        P02_P01 = comp_codes.stagpressnormshock(M_1)
        P02_P0inf = P01_P0inf*P02_P01
        P01_p1 = self._isen_flow_ratios("P", M_1)
        p2_P02 = 1./self._isen_flow_ratios("P", M_2)
        p2_p1 = P01_p1*P02_P01*p2_P02
        if not opt:
            print(f"Mach Number Behind Cowl Shock: {self.M_2:.4f}")
            print(f"Compression Ratio Through Cowl Shock: {p2_p1:.4f}")
            print(f"Ratio of Stagnation Pressure Behind Cowl Shock to Freestream: {P02_P0inf:.4f}\n")
        return P02_P0inf


    def min_P0_loss_3D(self):
        cone_angle_range = np.deg2rad(np.linspace(10., 30.))
        P0_drop = np.zeros(len(cone_angle_range))
        for i in range(len(cone_angle_range)):
            P0_drop[i] = self.taylor_maccoll(cone_angle_range[i], opt=True)
        plt.figure(3)
        plt.plot(cone_angle_range*180./np.pi, P0_drop, color='k', label='3D Solution')
        plt.xlabel('Cone Angle, deg')
        plt.ylabel(r'$\frac{P_{0_2}}{P_{0_\infty}}$')
        plt.xlim(1, 30)
        plt.xticks(np.arange(0, 31, 5))
        plt.legend()
        return cone_angle_range[np.nanargmax(P0_drop)], np.max(P0_drop)

    def min_P0_loss_2D(self):
        cone_angle_range = np.deg2rad(np.linspace(1., 16.))
        P0_drop = np.zeros(len(cone_angle_range))
        for i in range(len(cone_angle_range)):
            P0_drop[i] = self.twoD_solution(cone_angle_range[i], opt=True)
        plt.figure(3)
        plt.plot(cone_angle_range*180./np.pi, P0_drop, color='k', linestyle='--', label='2D Solution')
        return cone_angle_range[np.nanargmax(P0_drop)], np.max(P0_drop)


if __name__ == "__main__":
    plt.close('all')
    M_inf = 1.7
    beta_init = np.deg2rad(45.)
    cone_angle = np.deg2rad(15.)
    case = ConicalShock(M_inf, beta_init, N=400)
    print("\n--------------------------------\n| Taylor-Maccoll Solution (3D) |\n--------------------------------\n")
    P0_drop = case.taylor_maccoll(cone_angle)
    print("\n---------------------------------\n| Theta-Beta-Mach Solution (2D) |\n---------------------------------\n")
    case.twoD_solution(cone_angle)
    cone_min_pressure_loss2d, P02_P0inf_2d = case.min_P0_loss_2D()
    print(f"Cone Angle for Minimum Loss of Stagnation Pressure (2-D): {np.rad2deg(cone_min_pressure_loss2d):.4f}")
    print(f"Percent Loss in Stagnation Pressure (2-D): {(1 - P02_P0inf_2d)*100:.4f} %")
    cone_min_pressure_loss3d, P02_P0inf_3d = case.min_P0_loss_3D()
    print(f"Cone Angle for Minimum Loss of Stagnation Pressure (3-D): {np.rad2deg(cone_min_pressure_loss3d):.4f}")
    print(f"Ratio of Stagnation Pressure Behind Cowl Shock to Freestream (3-D): {(1 - P02_P0inf_3d)*100:.4f} %")
