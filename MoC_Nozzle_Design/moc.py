#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functions and a class for designing nozzles using the Method of Characteristics (MoC).

Functions:
    prandtl_meyer(nu, gamma, M): Computes the Mach number for a given Prandtl-Meyer angle.
    _nu_M(M, gamma): Computes the Prandtl-Meyer function for a given Mach number.
    _dnu_dM(M, gamma): Computes the derivative of the Prandtl-Meyer function with respect to Mach number.
    A_Astar(A_Astar, gamma, M): Computes the Mach number for a given area ratio.
    _dF_dM(M, gamma): Computes the derivative of the area ratio function with respect to Mach number.
    _F_M(M, gamma, A_Astar): Computes the area ratio function for a given Mach number.
    _expansion_ratio(M, gamma): Computes the expansion ratio for a given Mach number.

Classes:
    MethodOfCharacteristics: Class for designing nozzles using the Method of Characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt


class MethodOfCharacteristics:
    """
    Class for designing nozzles using the Method of Characteristics (MoC).

    Attributes:
        gamma (float): Specific heat ratio.
        N (int): Number of characteristic lines.
        theta_max (float): Maximum flow angle.
        d_theta (float): Incremental flow angle.
        theta_w (ndarray): Flow angles at the wall.
        nu_w (ndarray): Prandtl-Meyer function values at the wall.
        M_w (ndarray): Mach numbers at the wall.
        mu_w (ndarray): Mach angles at the wall.
        x_w (ndarray): x-coordinates of the wall.
        y_w (ndarray): y-coordinates of the wall.
        x_inner (ndarray): x-coordinates of the inner points.
        y_inner (ndarray): y-coordinates of the inner points.
        M_inner (ndarray): Mach numbers at the inner points.
        theta_inner (ndarray): Flow angles at the inner points.
        x_s (ndarray): x-coordinates of the straightening section.
        y_s (ndarray): y-coordinates of the straightening section.
        M_s (ndarray): Mach numbers at the straightening section.
        nu_s (ndarray): Prandtl-Meyer function values at the straightening section.
        x_clout (ndarray): x-coordinates of the centerline.
        y_clout (ndarray): y-coordinates of the centerline.
        M_clout (ndarray): Mach numbers at the centerline.
        nu_clout (ndarray): Prandtl-Meyer function values at the centerline.
        x_bell (ndarray): x-coordinates of the bell nozzle contour.
        y_bell (ndarray): y-coordinates of the bell nozzle contour.
        P (float): Bell nozzle coefficient P.
        S (float): Bell nozzle coefficient S.
        Q (float): Bell nozzle coefficient Q.
        T (float): Bell nozzle coefficient T.
        x_n (float): x-coordinate of the nozzle exit.
        y_n (float): y-coordinate of the nozzle exit.
    """
    def __init__(self, gamma=1.4):
        """
        Initializes the MethodOfCharacteristics class with the given specific heat ratio.

        Args:
            gamma (float): Specific heat ratio. Must be greater than 1.
        """
        if gamma <= 1:
            raise ValueError("Specific heat ratio (gamma) must be greater than 1.")
        self.gamma = gamma

    def prandtl_meyer(self, nu, M):
        """
        Computes the Mach number for a given Prandtl-Meyer angle using an iterative method.

        Args:
            nu (float): Prandtl-Meyer angle.
            M (float): Initial guess for the Mach number. Must be greater than 1.

        Returns:
            float: Mach number corresponding to the given Prandtl-Meyer angle.
        """
        if self.gamma <= 1:
            raise ValueError("Specific heat ratio (gamma) must be greater than 1.")
        if M <= 1:
            raise ValueError("Initial guess for Mach number (M) must be greater than 1.")
        
        error = 1e-2
        max_iterations = 1000
        iterations = 0
        
        while error > 1e-12:
            if iterations > max_iterations:
                raise RuntimeError("Maximum iterations reached in prandtl_meyer function.")
            
            nu_M = self._nu_M(M)
            dnu_dM = self._dnu_dM(M)
            Mp1 = M + (nu - nu_M) / dnu_dM
            error = np.abs((nu - nu_M) / nu_M)
            M = Mp1
            iterations += 1
        
        return M

    def _common_calculations(self, M):
        """
        Computes common values used in multiple functions.

        Args:
            M (float): Mach number.

        Returns:
            tuple: gp1, gm1, mach_squared_minus_one
        """
        gp1 = self.gamma + 1.
        gm1 = self.gamma - 1.
        mach_squared_minus_one = M * M - 1.
        return gp1, gm1, mach_squared_minus_one

    def _nu_M(self, M):
        """
        Computes the Prandtl-Meyer function for a given Mach number.

        Args:
            M (float): Mach number. Must be greater than 1.

        Returns:
            float: Prandtl-Meyer function value.
        """
        if M <= 1:
            raise ValueError("Mach number (M) must be greater than 1.")
        
        gp1, gm1, mach_squared_minus_one = self._common_calculations(M)
        C1 = np.sqrt(gp1 / gm1)
        C2 = np.arctan2(np.sqrt(gm1 * mach_squared_minus_one), np.sqrt(gp1))
        C3 = np.arctan2(np.sqrt(mach_squared_minus_one), 1.)
        nu_M = C1 * C2 - C3
        return nu_M

    def _dnu_dM(self, M):
        """
        Computes the derivative of the Prandtl-Meyer function with respect to Mach number.

        Args:
            M (float): Mach number. Must be greater than 1.

        Returns:
            float: Derivative of the Prandtl-Meyer function.
        """
        if M <= 1:
            raise ValueError("Mach number (M) must be greater than 1.")
        
        gp1, gm1, mach_squared_minus_one = self._common_calculations(M)
        C1 = 1. / M
        C2 = np.sqrt(mach_squared_minus_one)
        C3 = 1. + gm1 * M * M / 2.
        dnu_dM = C1 * C2 / C3
        return dnu_dM

    def A_Astar(self, A_Astar, M):
        """
        Computes the Mach number for a given area ratio using an iterative method.

        Args:
            A_Astar (float): Area ratio.
            M (float): Initial guess for the Mach number. Must be greater than 1.

        Returns:
            float: Mach number corresponding to the given area ratio.
        """
        if self.gamma <= 1:
            raise ValueError("Specific heat ratio (gamma) must be greater than 1.")
        if M <= 1:
            raise ValueError("Initial guess for Mach number (M) must be greater than 1.")
        
        error = 100.
        max_iterations = 1000
        iterations = 0
        
        while error > 1e-12:
            if iterations > max_iterations:
                raise RuntimeError("Maximum iterations reached in A_Astar function.")
            
            F_M = self._F_M(M, A_Astar)
            dF_dM = self._dF_dM(M)
            Mp1 = M - F_M / dF_dM
            error = np.abs(F_M) / A_Astar
            M = Mp1
            iterations += 1
        
        return M

    def _dF_dM(self, M):
        """
        Computes the derivative of the area ratio function with respect to Mach number.

        Args:
            M (float): Mach number.

        Returns:
            float: Derivative of the area ratio function.
        """
        gp1, gm1, mach_squared_minus_one = self._common_calculations(M)
        C1 = np.power(2., (1. - 3. * self.gamma) / (2. - 2. * self.gamma))
        C2num = mach_squared_minus_one
        C2denom = M * M * (2. + M * M * (self.gamma - 1.))
        C2 = C2num / C2denom
        C3exp = (self.gamma + 1.) / (2 * (self.gamma - 1.))
        C3num = (1. + (self.gamma - 1.) * M * M / 2.)
        C3denom = self.gamma + 1.
        C3 = np.power((C3num / C3denom), C3exp)
        dF_dM = C1 * C2 * C3
        return dF_dM

    def _F_M(self, M, A_Astar):
        """
        Computes the area ratio function for a given Mach number.

        Args:
            M (float): Mach number.
            A_Astar (float): Area ratio.

        Returns:
            float: Area ratio function value.
        """
        gp1, gm1, mach_squared_minus_one = self._common_calculations(M)
        C1 = 1. / M
        C2 = (2. / (self.gamma + 1.))
        C3 = (1. + (self.gamma - 1.) * M * M / 2.)
        E1 = (self.gamma + 1.) / (2. * (self.gamma - 1.))
        C4 = A_Astar
        F_M = C1 * np.power(C2 * C3, E1) - C4
        return F_M

    def _expansion_ratio(self, M):
        """
        Computes the expansion ratio for a given Mach number.

        Args:
            M (float): Mach number.

        Returns:
            float: Expansion ratio.
        """
        C1 = 1./M
        C2 = (2./(self.gamma + 1.))
        C3 = (1. + (self.gamma - 1.)*M*M/2.)
        E1 = (self.gamma + 1.)/(2.*(self.gamma - 1.))
        eps = C1*np.power(C2*C3, E1)
        return eps

    def _compat_constant(self, M, theta, left=True):
        """
        Computes the compatibility constant for a given Mach number and flow angle.

        Args:
            M (float): Mach number.
            theta (float): Flow angle.
            left (bool): Whether the characteristic is left-running.

        Returns:
            float: Compatibility constant.
        """
        nu = self._nu_M(M)
        if not left:
            K = theta + nu
        else:
            K = theta - nu
        return K

    def _nozzle_properties(self, D_t, theta_e, M_e):
        """
        Computes the nozzle properties for a given throat diameter, exit angle, and exit Mach number.

        Args:
            D_t (float): Throat diameter.
            theta_e (float): Exit angle.
            M_e (float): Exit Mach number.

        Returns:
            tuple: Expansion ratio, x-coordinate of the nozzle exit, y-coordinate of the nozzle exit, nozzle length.
        """
        eps = self._expansion_ratio(M_e)
        R_t = D_t/2.
        R_1 = 1.5*R_t
        x_n = R_1*np.sin(theta_e)
        y_n = R_t + R_1*(1 - np.cos(theta_e))
        num_1 = 0.5*D_t*(eps - 1.)
        num_2 = R_1*(1./np.cos(theta_e) - 1.)
        L_nozzle = (num_1 + num_2)/np.tan(theta_e)
        return eps, x_n, y_n, L_nozzle

    def _bell_coefficients(self, xp_e, yp_e, theta_n, theta_e):
        """
        Computes the coefficients for the bell nozzle contour.

        Args:
            xp_e (float): x-coordinate of the parabolic fit exit.
            yp_e (float): y-coordinate of the parabolic fit exit.
            theta_n (float): Nozzle angle.
            theta_e (float): Exit angle.

        Returns:
            tuple: Bell nozzle coefficients P, S, Q, T.
        """
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
        """
        Computes the properties at point 3 for a given compatibility constants K1 and K2.

        Args:
            K1 (float): Compatibility constant 1.
            K2 (float): Compatibility constant 2.

        Returns:
            tuple: Flow angle, Prandtl-Meyer function value, Mach number at point 3.
        """
        theta_3 = (K1 + K2)/2.
        nu_3 = (K1 - K2)/2.
        M_3 = self.prandtl_meyer(nu_3, 1.5)
        return theta_3, nu_3, M_3

    def _mach_angle(self, M):
        """
        Computes the Mach angle for a given Mach number.

        Args:
            M (float): Mach number.

        Returns:
            list: Mach angles.
        """
        mu = []
        for m in M:
            mu.append(np.arcsin(1./m))
        return mu

    def _char_slope(self, theta, mu):
        """
        Computes the characteristic slopes for a given flow angle and Mach angle.

        Args:
            theta (list): Flow angles.
            mu (list): Mach angles.

        Returns:
            tuple: Characteristic slopes m_cm and m_cp.
        """
        m_cm = 0.5*((theta[0] - mu[0]) + (theta[2] - mu[2]))
        m_cp = 0.5*((theta[1] + mu[1]) + (theta[2] + mu[2]))
        return m_cm, m_cp

    def _point3_xy(self, m_cm, m_cp, x, y):
        """
        Computes the coordinates of point 3 for given characteristic slopes and coordinates.

        Args:
            m_cm (float): Characteristic slope m_cm.
            m_cp (float): Characteristic slope m_cp.
            x (list): x-coordinates.
            y (list): y-coordinates.

        Returns:
            tuple: x and y coordinates of point 3.
        """
        x1, x2 = x
        y1, y2 = y
        t_mcm = np.tan(m_cm)
        t_mcp = np.tan(m_cp)
        x3 = (x1*t_mcm - x2*t_mcp + y2 - y1)/(t_mcm - t_mcp)
        y3 = (t_mcm*t_mcp*(x1 - x2) + t_mcm*y2 - t_mcp*y1)/(t_mcm - t_mcp)
        return x3, y3

    def _unit_internal(self, M, theta, xy1, xy2):
        """
        Computes the internal unit properties for given Mach numbers, flow angles, and coordinates.

        Args:
            M (list): Mach numbers.
            theta (list): Flow angles.
            xy1 (list): Coordinates 1.
            xy2 (list): Coordinates 2.

        Returns:
            tuple: x and y coordinates, Mach number, and flow angle at point 3.
        """
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
        """
        Computes the straightening section of the nozzle.

        Returns:
            tuple: x and y coordinates, Mach numbers, and Prandtl-Meyer function values of the straightening section.
        """
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
            nu_f = self._nu_M(M_f)
            mu_f = self._mach_angle([M_f])[0]
            nu_s[j + 1] = theta_1 - theta_f + nu_f
            M_s[j + 1] = self.prandtl_meyer(nu_s[j + 1], M_1)
            mu_s = self._mach_angle([M_s[j + 1]])[0]
            m_cm, m_cp = self._char_slope([0., theta_f, theta_1], [0., mu_f, mu_s])
            x_s[j + 1], y_s[j + 1] = self._wall_xy(m_cp, theta_1, [x_1, x_f], [y_1, y_f])
            plt.plot([x_f*100., x_s[j + 1]*100.], [y_f*100., y_s[j + 1]*100.])
        return x_s, y_s, M_s, nu_s

    def _unit_centerline(self, nu_w, theta_w, M_w, mu_w):
        """
        Computes the properties at the centerline for given wall properties.

        Args:
            nu_w (float): Prandtl-Meyer function value at the wall.
            theta_w (float): Flow angle at the wall.
            M_w (float): Mach number at the wall.
            mu_w (float): Mach angle at the wall.

        Returns:
            tuple: Prandtl-Meyer function value, Mach number, Mach angle, and characteristic slope at the centerline.
        """
        nu_cl = theta_w + nu_w
        M_cl = self.prandtl_meyer(nu_cl, M_w)
        mu_cl = self._mach_angle([M_cl])[0]
        m_cm, m_cp = self._char_slope([theta_w, 0., 0.], [mu_w, 0., mu_cl])
        return nu_cl, M_cl, mu_cl, m_cm

    def _wall_xy(self, m_cp, theta_s, x, y):
        """
        Computes the coordinates of the wall for given characteristic slopes and coordinates.

        Args:
            m_cp (float): Characteristic slope m_cp.
            theta_s (float): Flow angle at the straightening section.
            x (list): x-coordinates.
            y (list): y-coordinates.

        Returns:
            tuple: x and y coordinates of the wall.
        """
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
        """
        Computes the expansion grid for the nozzle.

        Returns:
            tuple: x and y coordinates, Mach numbers, and flow angles of the inner points.
        """
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
        x_clp1 = self.x_w[0]  # Initialize x_clp1
        y_clp1 = self.y_w[0]  # Initialize y_clp1
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
            x_cl[j] = -y_clp1 / np.tan(m_cm) + x_clp1
            y_cl[j] = 0.
            theta_cl[j] = 0.
            # plt.scatter(x_cl[j], y_cl[j])
            plt.plot([x_clp1 * 100, x_cl[j] * 100], [y_clp1 * 100, y_cl[j] * 100])
            if self.N - 1 - j > 0:
                x_inner[j + 1, j] = x_cl[j]
                y_inner[j + 1, j] = y_cl[j]
                M_inner[j + 1, j] = M_cl[j]
                theta_inner[j + 1, j] = theta_cl[j]
                for i in range(j, self.N - 1):
                    x_inner[j + 1, i + 1], y_inner[j + 1, i + 1], M_inner[j + 1, i + 1], theta_inner[j + 1, i + 1] = self._unit_internal(
                        [M_inner[j, i + 1], M_inner[j + 1, i]],
                        [theta_inner[j, i + 1], theta_inner[j + 1, i]],
                        [x_inner[j, i + 1], y_inner[j, i + 1]],
                        [x_inner[j + 1, i], y_inner[j + 1, i]]
                    )
                    # plt.scatter(x_inner[j + 1, i + 1], y_inner[j + 1, i + 1])
                    plt.plot([x_inner[j + 1, i] * 100, x_inner[j + 1, i + 1] * 100], [y_inner[j + 1, i] * 100, y_inner[j + 1, i + 1] * 100])
                    plt.plot([x_inner[j, i + 1] * 100, x_inner[j + 1, i + 1] * 100], [y_inner[j, i + 1] * 100, y_inner[j + 1, i + 1] * 100])
                nu_clp1 = self._nu_M(M_inner[j + 1, j + 1])
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
        """
        Designs a nozzle using the Method of Characteristics.

        Args:
            M_exit (float): Exit Mach number.
            D_t (float): Throat diameter.
            gamma (float): Specific heat ratio. Default is 1.4.
            N (int): Number of characteristic lines. Default is 10.
            bell (bool): Whether to design a bell nozzle. Default is True.
            R_c (float): Radius of curvature at the throat. Default is 0.
            theta_e (float): Exit angle. Default is None.
        """
        self.N = N
        self.gamma = gamma
        nu_e = self._nu_M(M_exit)
        self.theta_max = nu_e/2.
        self.d_theta = self.theta_max/self.N
        self.theta_w = np.arange(self.d_theta, self.theta_max + 1e-5, self.d_theta)
        self.nu_w = self.theta_w
        self.M_w = [self.prandtl_meyer(n, 1.2) for n in self.nu_w]
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
        plt.legend(loc='upper right')
        self._mach_contours()

    def _nozzle_bell_3d(self, M_exit, D_t):
        """
        Designs a bell nozzle contour.

        Args:
            M_exit (float): Exit Mach number.
            D_t (float): Throat diameter.
        """
        theta_n = self.theta_max
        theta_exit = 0.
        eps = self._expansion_ratio(M_exit)
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
        """
        Plots the Mach number contours along the nozzle.
        """
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
            M_AAstar[i] = self.A_Astar(A_Astar_profile[i], 1.2)
        plt.figure()
        plt.scatter(wall_x*100, wall_M, edgecolors='k', facecolors='none')
        plt.plot(wall_x*100, wall_M, color='k', linestyle='--', label='$M_\mathrm{wall}$')
        plt.scatter(center_x*100, center_M, edgecolors='g', facecolors='none')
        plt.plot(center_x*100, center_M, color='g', linestyle='--', label='$M_\mathrm{center}$')
        plt.scatter(wall_x*100., M_AAstar, edgecolors='b', facecolors='none')
        plt.plot(wall_x*100., M_AAstar, color='b', linestyle='--', label='$M_{A/A^*}$')
        plt.grid(visible=True)
        plt.xlabel('$x$, cm')
        plt.ylabel('$M$')
        plt.xlim((self.x_w[0]*100., self.x_clout[-1]*100.))
        plt.ylim((0., None))
        plt.legend(loc='lower left')
        plt.show()



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
