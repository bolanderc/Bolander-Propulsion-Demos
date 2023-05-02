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

def A_Astar(A_Astar, gamma, M):
    error = 100.
    while error > 1e-12:
        F_M = _F_M(M, gamma, A_Astar)
        dF_dM = _dF_dM(M, gamma)
        Mp1 = M - F_M/dF_dM
        error = np.abs(F_M)/A_Astar
        M = Mp1
    return M

def _dF_dM(M, gamma):
    C1 = np.power(2., (1. - 3.*gamma)/(2. - 2.*gamma))
    C2num = (M*M - 1.)
    C2denom = M*M*(2. + M*M*(gamma - 1.))
    C2 = C2num/C2denom
    C3exp = (gamma + 1.)/(2*(gamma - 1.))
    C3num = (1. + (gamma - 1.)*M*M/2.)
    C3denom = gamma + 1.
    C3 = np.power((C3num/C3denom), C3exp)
    dF_dM = C1*C2*C3
    return dF_dM

def _F_M(M, gamma, A_Astar):
    C1 = 1./M
    C2 = (2./(gamma + 1.))
    C3 = (1. + (gamma - 1.)*M*M/2.)
    E1 = (gamma + 1.)/(2.*(gamma - 1.))
    C4 = A_Astar
    F_M = C1*np.power(C2*C3, E1) - C4
    return F_M

def _expansion_ratio(M, gamma):
    C1 = 1./M
    C2 = (2./(gamma + 1.))
    C3 = (1. + (gamma - 1.)*M*M/2.)
    E1 = (gamma + 1.)/(2.*(gamma - 1.))
    eps = C1*np.power(C2*C3, E1)
    return eps

class MethodOfCharacteristics:
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    def _compat_constant(self, M, theta, left=True):
        nu = _nu_M(M, self.gamma)
        if not left:
            K = theta + nu
        else:
            K = theta - nu
        return K

    def _nozzle_properties(self, D_t, theta_e, M_e):
        eps = _expansion_ratio(M_e, self.gamma)
        R_t = D_t/2.
        R_1 = 1.5*R_t
        x_n = R_1*np.sin(theta_e)
        y_n = R_t + R_1*(1 - np.cos(theta_e))
        num_1 = 0.5*D_t*(eps - 1.)
        num_2 = R_1*(1./np.cos(theta_e) - 1.)
        L_nozzle = (num_1 + num_2)/np.tan(theta_e)
        return eps, x_n, y_n, L_nozzle

    def _bell_coefficients(self, xp_e, yp_e, theta_n, theta_e):
        t_n = np.tan(theta_n)
        t_e = np.tan(theta_e)
        num = yp_e*(t_n + t_e) - 2.*xp_e*t_e*t_n
        denom = 2.*yp_e - xp_e*t_n - xp_e*t_e
        P = num/denom
        num = (yp_e - P*xp_e)**2*(t_n - P)
        denom = xp_e*t_n - yp_e
        S = num/denom
        Q = -S/(2.*(t_n - P))
        T = Q*Q
        return P, S, Q, T

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

    def _unit_internal(self, M, theta, xy1, xy2):
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

    def _straightening_section(self):
        self.theta_s = np.arange(self.theta_max, -self.d_theta, -self.d_theta)
        x_s = np.zeros(self.N + 1)
        y_s = np.zeros_like(x_s)
        M_s = np.zeros_like(x_s)
        nu_s = np.zeros_like(x_s)
        x_s[0] = self.x_w[-1]
        y_s[0] = self.y_w[-1]
        M_s[0] = self.M_w[-1]
        nu_s[0] = self.nu_w[-1]
        for j in range(self.N):
            x_f = self.x_inner[j + 1, -1]
            y_f = self.y_inner[j + 1, -1]
            theta_f = self.theta_inner[j + 1, -1]
            M_f = self.M_inner[j + 1, -1]
            x_1 = x_s[j]
            y_1 = y_s[j]
            theta_1 = (self.theta_s[j] + self.theta_s[j + 1])/2.
            M_1 = M_s[j]
            nu_f = _nu_M(M_f, self.gamma)
            mu_f = self._mach_angle([M_f])[0]
            nu_s[j + 1] = theta_1 - theta_f + nu_f
            M_s[j + 1] = prandtl_meyer(nu_s[j + 1], self.gamma, M_1)
            mu_s = self._mach_angle([M_s[j + 1]])[0]
            m_cm, m_cp = self._char_slope([0., theta_f, theta_1], [0., mu_f, mu_s])
            x_s[j + 1], y_s[j + 1] = self._wall_xy(m_cp, theta_1, [x_1, x_f], [y_1, y_f])
            plt.plot([x_f*100., x_s[j + 1]*100.], [y_f*100., y_s[j + 1]*100.])
        return x_s, y_s, M_s, nu_s

    def _unit_centerline(self, nu_w, theta_w, M_w, mu_w):
        nu_cl = theta_w + nu_w
        M_cl = prandtl_meyer(nu_cl, self.gamma, M_w)
        mu_cl = self._mach_angle([M_cl])[0]
        m_cm, m_cp = self._char_slope([theta_w, 0., 0.], [mu_w, 0., mu_cl])
        return nu_cl, M_cl, mu_cl, m_cm

    def _wall_xy(self, m_cp, theta_s, x, y):
        x_wm1, xf = x
        y_wm1, yf = y
        t_mcp = np.tan(m_cp)
        t_theta = np.tan(theta_s)
        denom = t_theta - t_mcp
        x_num = x_wm1*t_theta - xf*t_mcp + (yf - y_wm1)
        y_num = t_theta*t_mcp*(x_wm1 - xf) + t_theta*yf - t_mcp*y_wm1
        x_s = x_num/denom
        y_s = y_num/denom
        return x_s, y_s

    def _expansion_grid(self):
        nu_cl = np.zeros_like(self.nu_w)
        M_cl = np.zeros_like(self.M_w)
        mu_cl = np.zeros_like(self.mu_w)
        theta_cl = np.zeros_like(self.theta_w)
        x_cl = np.zeros_like(self.x_w)
        y_cl = np.zeros_like(self.y_w)
        nu_clp1 = self.nu_w[0]
        theta_clp1 = self.theta_w[0]
        M_clp1 = self.M_w[0]
        mu_clp1 = self.mu_w[0]
        x_clp1 = self.x_w[0]
        y_clp1 = self.y_w[0]
        x_inner = np.zeros((self.N + 1, self.N))
        y_inner = np.zeros_like(x_inner)
        M_inner = np.zeros_like(x_inner)
        theta_inner = np.zeros_like(x_inner)
        x_inner[0, 1:] = self.x_w[1:]
        y_inner[0, 1:] = self.y_w[1:]
        M_inner[0, 1:] = self.M_w[1:]
        theta_inner[0, 1:] = self.theta_w[1:]
        for j in range(self.N):
            nu_cl[j], M_cl[j], mu_cl[j], m_cm = self._unit_centerline(nu_clp1, theta_clp1, M_clp1, mu_clp1)
            x_cl[j] = -y_clp1/np.tan(m_cm) + x_clp1
            y_cl[j] = 0.
            theta_cl[j] = 0.
            # plt.scatter(x_cl[j], y_cl[j])
            plt.plot([x_clp1*100, x_cl[j]*100], [y_clp1*100, y_cl[j]*100])
            if self.N - 1 - j > 0:
                x_inner[j + 1, j] = x_cl[j]
                y_inner[j + 1, j] = y_cl[j]
                M_inner[j + 1, j] = M_cl[j]
                theta_inner[j + 1, j] = theta_cl[j]
                for i in range(j, self.N - 1):
                    x_inner[j + 1, i + 1], y_inner[j + 1, i + 1], M_inner[j + 1, i + 1], theta_inner[j + 1, i + 1] = self._unit_internal([M_inner[j, i + 1], M_inner[j + 1, i]], [theta_inner[j, i + 1], theta_inner[j + 1, i]], [x_inner[j, i + 1], y_inner[j, i + 1]], [x_inner[j + 1, i], y_inner[j + 1, i]])
                    # plt.scatter(x_inner[j + 1, i + 1], y_inner[j + 1, i + 1])
                    plt.plot([x_inner[j + 1, i]*100, x_inner[j + 1, i + 1]*100], [y_inner[j + 1, i]*100, y_inner[j + 1, i + 1]*100])
                    plt.plot([x_inner[j, i + 1]*100, x_inner[j + 1, i + 1]*100], [y_inner[j, i + 1]*100, y_inner[j + 1, i + 1]*100])
                nu_clp1 = _nu_M(M_inner[j + 1, j + 1], self.gamma)
                mu_clp1 = self._mach_angle([M_inner[j + 1, j + 1]])[0]
                theta_clp1 = theta_inner[j + 1, j + 1]
                M_clp1 = M_inner[j + 1, j + 1]
                x_clp1 = x_inner[j + 1, j + 1]
                y_clp1 = y_inner[j + 1, j + 1]
            else:
                x_inner[-1, -1] = x_cl[j]
                y_inner[-1, -1] = y_cl[j]
                M_inner[-1, -1] = M_cl[j]
                theta_inner[-1, -1] = 0.
        return x_inner, y_inner, M_inner, theta_inner

    def moc_nozzle_design(self, M_exit, D_t, gamma=1.4, N=10, bell=True, R_c=0.,
                          theta_e=None):
        self.N = N
        self.gamma = gamma
        nu_e = _nu_M(M_exit, self.gamma)
        self.theta_max = nu_e/2.
        self.d_theta = self.theta_max/self.N
        self.theta_w = np.arange(self.d_theta, self.theta_max + 1e-5, self.d_theta)
        self.nu_w = self.theta_w
        self.M_w = [prandtl_meyer(n, self.gamma, 1.2) for n in self.nu_w]
        self.mu_w = self._mach_angle(self.M_w)
        self.x_w = R_c*np.sin(self.theta_w)
        self.y_w = 0.5*D_t + R_c*(1. - np.cos(self.theta_w))
        plt.figure()
        plt.plot(self.x_w*100, self.y_w*100, color='k', linewidth=3)
        plt.xlabel('$x$, cm')
        plt.ylabel('$y$, cm')
        self.x_inner, self.y_inner, self.M_inner, self.theta_inner = self._expansion_grid()
        self.x_s, self.y_s, self.M_s, self.nu_s = self._straightening_section()
        plt.plot(self.x_s*100, self.y_s*100, color='k', linewidth=3, label='MoC Contour')
        self.x_clout = np.zeros(self.N)
        self.y_clout = np.zeros_like(self.x_clout)
        self.M_clout = np.zeros_like(self.x_clout)
        self.nu_clout = np.zeros_like(self.x_clout)
        for i in range(self.N):
            mu_s = self._mach_angle([self.M_s[i + 1]])
            self.nu_clout[i], self.M_clout[i], mu_clout, m_cm = self._unit_centerline(self.nu_s[i + 1], self.theta_s[i + 1], self.M_s[i + 1], mu_s)
            self.x_clout[i] = -self.y_s[i + 1]/np.tan(m_cm) + self.x_s[i + 1]
            # plt.scatter(self.x_clout[i], 0.)
            plt.plot([self.x_s[i + 1]*100, self.x_clout[i]*100], [self.y_s[i + 1]*100, 0.])
        if bell:
            self._nozzle_bell_3d(M_exit, D_t)
        plt.legend()
        self._mach_contours()

    def _nozzle_bell_3d(self, M_exit, D_t):
        theta_n = self.theta_max
        theta_exit = 0.
        eps = _expansion_ratio(M_exit, self.gamma)
        self.x_n = self.x_w[-1]
        self.y_n = self.y_w[-1]
        L_nozzle = self.x_s[-1]
        xp_exit = L_nozzle - self.x_n
        yp_exit = self.y_s[-1] - self.y_n
        self.P, self.S, self.Q, self.T = self._bell_coefficients(xp_exit, yp_exit, theta_n, theta_exit)
        xp = np.linspace(0., xp_exit)
        yp = self.P*xp + self.Q + (self.S*xp + self.T)**0.5
        self.x_bell = xp + self.x_n
        self.y_bell = yp + self.y_n
        plt.plot(self.x_bell*100, self.y_bell*100, color='r', linewidth=3, linestyle='--', label='Parabolic Fit Contour')

    def _mach_contours(self):
        wall_x = np.concatenate((self.x_w, self.x_s[1:]))
        wall_M = np.concatenate((self.M_w, self.M_s[1:]))
        inner_x = np.diagonal(self.x_inner, offset=-1)
        inner_M = np.diagonal(self.M_inner, offset=-1)
        center_x = np.concatenate((inner_x, self.x_clout))
        center_M = np.concatenate((inner_M, self.M_clout))
        y_profile = np.concatenate((self.y_w, self.y_s[1:]))
        A_Astar_profile = y_profile/y_profile[0]
        M_AAstar = np.zeros(len(A_Astar_profile))
        for i in range(len(A_Astar_profile)):
            M_AAstar[i] = A_Astar(A_Astar_profile[i], self.gamma, 1.2)
        plt.figure()
        plt.scatter(wall_x*100, wall_M, edgecolors='k', facecolors='none')
        plt.plot(wall_x*100, wall_M, color='k', linestyle='--', label='$M_\mathrm{wall}$')
        plt.scatter(center_x*100, center_M, edgecolors='g', facecolors='none')
        plt.plot(center_x*100, center_M, color='g', linestyle='--', label='$M_\mathrm{center}$')
        plt.scatter(wall_x*100., M_AAstar, edgecolors='b', facecolors='none')
        plt.plot(wall_x*100., M_AAstar, color='b', linestyle='--', label='$M_{A/A^*}$')
        plt.grid(b=True)
        plt.xlabel('$x$, cm')
        plt.ylabel('$M$')
        plt.xlim((self.x_w[0]*100., self.x_clout[-1]*100.))
        plt.ylim((0., None))
        plt.legend()



if __name__ == "__main__":
    plt.close('all')
    case = MethodOfCharacteristics()
    N = 50
    M_exit = 2.0
    D_t = 0.02
    case.moc_nozzle_design(M_exit, D_t, N=N, gamma=1.4)
    print("Case 1: M=2.0, gamma = 1.4, N = 50, R_c = 0")
    print(f"P = {case.P:5f},\tS = {case.S:5f},\tQ = {case.Q:5f},\tT = {case.T:5f}")
    print(f"x_n = {case.x_n*100:5f} cm,\t y_n = {case.y_n*100:5f} cm\n")
    M_exit = 2.0
    D_t = 0.02
    case.moc_nozzle_design(M_exit, D_t, N=N, gamma=1.2)
    print("Case 2: M=2.0, gamma = 1.2, N = 50, R_c = 0")
    print(f"P = {case.P:5f},\tS = {case.S:5f},\tQ = {case.Q:5f},\tT = {case.T:5f}")
    print(f"x_n = {case.x_n*100:5f} cm,\t y_n = {case.y_n*100:5f} cm\n")
    M_exit = 2.0
    D_t = 0.02
    case.moc_nozzle_design(M_exit, D_t, N=N, gamma=1.4, R_c=1.5*D_t/2.)
    print("Case 3: M=2.0, gamma = 1.4, N = 50")
    print(f"P = {case.P:5f},\tS = {case.S:5f},\tQ = {case.Q:5f},\tT = {case.T:5f}")
    print(f"x_n = {case.x_n*100:5f} cm,\t y_n = {case.y_n*100:5f} cm\n")
    M_exit = 2.0
    D_t = 0.02
    case.moc_nozzle_design(M_exit, D_t, N=N, gamma=1.2, R_c=1.5*D_t/2.)
    print("Case 4: M=2.0, gamma = 1.2, N = 50")
    print(f"P = {case.P:5f},\tS = {case.S:5f},\tQ = {case.Q:5f},\tT = {case.T:5f}")
    print(f"x_n = {case.x_n*100:5f} cm,\t y_n = {case.y_n*100:5f} cm\n")
