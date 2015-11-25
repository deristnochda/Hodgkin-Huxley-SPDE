"""
File: spatialextneuron.py

Simulation of spatially extended stochastic neurons such as FHN or
Hodgkin-Huxley.

Provides Classes:
FitzHughNagumo
HodgkinHuxley
"""

import numpy as np
from numpy import random as rnd
from scipy import linalg as lng


class FitzHughNagumo:
    """
    Class for the FitzHugh-Nagumo neuron spatially extended with axon.
    
    dv/dt = D v_xx + kappa v (v-a) (1-v) - lam w + I
    dw/dt = epsilon (v - gamma w + b)

    where I is the applied current or input signal, e.g. a constant plus
    sigma * white noise.
    
    The init_state is [v0, w0] and
    D      : The diffusion constant,
    kappa  : scaling of nonlinearity,
    a      : second root of nonlinearity,
    I      : constant input current,
    lam    : coefficient of recovery variable,
    sigma  : the noise intensity,
    eps    : speed of the recovery variable,
    gammma : inverse slope of the recovery nullcline,
    b      : shift of the recovery nullcline,
    L      : length of the axon in space units,
    1/N    : size of the discrete grid,
    M      : number of realizations.
    """
    
    def __init__(self, init_state=[0,0], D=1., kappa=1., a=0.139, I=0.,
                 lam=1., sigma=0., eps=0.008, gamma=2.54, b=0., L=200,
                 N=400, M=1):
        """
        Initialize the neuron. The standard parameters are from Rinzel (1977, 
        Nonlinear diffusion) together with the equilibrium values as initial 
        conditions.
        """
        # Set parameters of FHN neuron
        self.params = np.asarray([D, kappa, a, I, lam, sigma, eps, gamma, b])
        # Set grid approximation constants
        self.L = L
        self.N = N
        self.dx = 1. * L / N
        self.axis_vector = np.arange(0, L + self.dx, self.dx)
        # Number of realizations
        self.M = M
        # Initialize state and time  
        self.state = np.asarray([init_state[0] * np.ones([N+1, M]),
                                 init_state[1] * np.ones([N+1, M])],
                                dtype='float')
        self.time_elapsed = 0
        # Matrix entries of discrete Neumann Laplacian
        self.bandedmatrix = np.vstack((np.ones(N+1),
                                       np.hstack((-1, -2 * np.ones(N-1), -1))))
        # Define dummy variable for dt-dependent coefficients
        self.time_params = []
        
    def init_time_scheme(self, dt):
        """
        Choice of appropriate scheme for time approximation. We use a semi-
        implicit Euler-Maruyama scheme with the taming suggested by Jentzen
        
        At the moment without taming!
        """
        [D, kappa, a, I, lam, sigma, eps, gamma, b] = self.params
        h = self.dx
        # Compute matrix for implicit Euler
        diffusion_constant = - D * dt / (h**2)
        self.bandedmatrix *= diffusion_constant
        self.bandedmatrix[1] += 1
        self.bandedmatrix[1,0] -= 0.5
        # Homogeneous Neumann at right boundary
        self.bandedmatrix[1,self.N] -= 0.5
        # Mixed boundary condition
        #self.bandedmatrix[1,self.N] -= 0.5 - diffusion_constant * h * 0.1
        # Compute coefficients used in each timestep
        self.time_params = np.asarray([2 * D * dt / h,
                                       sigma * np.sqrt(dt / h),
                                       1/(1 + dt * eps * gamma),
                                       dt * eps / (1 + dt * eps * gamma)],
                                      dtype='float')
        
    def timestep(self, dt, input_signal):
        """
        Compute the time evolution for one timestep of length dt
        """
        [kappa, a, I, lam, b] = self.params[[1,2,3,4,8]]
        [c1, c2, c3, c4] = self.time_params
        [v, w] = self.state
        rhs = (v + dt * (kappa * v * (1 - v) * (v - a) - lam * w + I)
               + c2 * np.vstack([rnd.normal(0, 1 / np.sqrt(2), [1, self.M]),
                                 rnd.normal(0, 1, [self.N - 1, self.M]),
                                 rnd.normal(0, 1 / np.sqrt(2), [1, self.M])]))
        # Input signal as Neumann boundary condition
        rhs[0] += c1 * input_signal
        # Modification of RHS to make discrete Neumann Laplacian symmetric
        rhs[0] *= 0.5
        rhs[self.N] *= 0.5
        # Semi-implicit Euler step for v
        self.state[0] = lng.solveh_banded(self.bandedmatrix, rhs,
                                          check_finite=False)
        # Implicit Euler step for w                                
        self.state[1] = c3 * w + c4 * (self.state[0] + b)
        self.time_elapsed += 1
        
    def linearized_timestep(self, dt, pulse_old, pulse_new):
        """
        Compute the linearized time evolution around the pulse solution for one
        timestep of length dt. Needs a reference pulse to compute the gradient.
        """
        [kappa, a, I, lam, b] = self.params[[1,2,3,4,8]]
        [c1, c2, c3, c4] = self.time_params
        [v_hat, w_hat] = pulse_old
        [v, w] = self.state - pulse_old
        rhs = (v + dt * (kappa * (2 * (1 + a) * v_hat - a - 3 * v_hat**2) * v
                         - lam * w)
               + c2 * np.vstack([rnd.normal(0, 1 / np.sqrt(2), [1, self.M]),
                                 rnd.normal(0, 1, [self.N - 1, self.M]),
                                 rnd.normal(0, 1 / np.sqrt(2), [1, self.M])]))
        # Modification of RHS to make discrete Neumann Laplacian symmetric
        rhs[0] *= 0.5
        rhs[self.N] *= 0.5
        # Semi-implicit Euler step for v
        self.state[0] = lng.solveh_banded(self.bandedmatrix, rhs,
                                          check_finite=False)
        # Implicit Euler step for w                                
        self.state[1] = c3 * w + c4 * self.state[0]
        self.state += pulse_new
        self.time_elapsed += 1     
        
    def area(self):
        """
        Compute the area below the traveling pulse
        """
        area = (self.dx * np.sum(self.state[0][1:self.N,:], axis=0) +
                self.dx * 0.5 * (self.state[0][0,:] + self.state[0,self.N,:]))
        return area
    
    def gradient_f(self):
        """
        Return the gradient of the nonlinearity at the current time
        """
        [kappa, a] = self.params[[1,2]]
        gradient = kappa * (2 * (1 + a) * self.state[0] - a
                            - 3 * self.state[0]**2)
        return gradient


class HodgkinHuxley():
    """
    Class for the spatially extended Hodgkin-Huxley type neuron.

    dV/dt = D V_xx - I_Na - I_K - I_leak + I
    
    where
    
    I_Na = g_Na m^3 h (V - E_Na),
    I_K = g_K n^4 (V - E_K),
    I_leak = g_leak (V - E_leak)
    
    and
    
    dx/dt = alpha_x (1 - x) - beta_x x, x = n, m, h.

    Some additional applied current, e.g. white noise, is modelled by I.
    Input signal is modelled through a Neumann boundary as an injected current.
    
    The init_state is [v0, n0, m0, h0] and
    f_leak     : leak current in uA/cm^2 as a function of v,
    f_Na       : sodium current in uA/cm^2 as a function of v, m, h,
    f_K        : potassium current in uA/cm^2 as a function of v, n,
    alpha_x    : opening rate of channel x as a function of v,
    beta_x     : closing rate of channel x as a function of v,
    sigma_curr : intensity of current noise,
    sigma_sub  : 0-1 switch for subunit noise,
    sigma_cond : 0-1 switch for conductance noise,
    rho_Na     : density of sodium channels per um^2,
    rho_K      : density of potassium channels per um^2,
    d          : diameter of the axon,
    R_i        : intracellular resistivity in kOhm cm,
    L          : length of the axon in cm,
    N          : spatial resolution of the grid,
    extension  : length of noiseless extension of axon in cm,
    dt         : time resolution in ms,
    M          : number of realizations.
    """
    
    def __init__(self,
                 init_state=[0.0, 0.053, 0.596, 0.318],
                 f_leak=lambda v: 0.3*(v-10.6),
                 f_Na=lambda v,m,h: 120.*(m**3)*h*(v-115.),
                 f_K=lambda v,n: 36.*(n**4)*(v+12.),
                 alpha_m=lambda v: 0.1*(v-25)/(1-np.exp(-0.1*(v-25))),
                 beta_m=lambda v: 4.*np.exp(v/(-18)),
                 alpha_h=lambda v: 0.07*np.exp(-0.05*v),
                 beta_h=lambda v: 1/(1+np.exp(-0.1*(v-30))),
                 alpha_n=lambda v: 0.01*(v-10)/(1-np.exp(-0.1*(v-10))),
                 beta_n=lambda v: 0.125*np.exp(v/(-80)),
                 sigma_curr=0.0,
                 sigma_sub=0,
                 sigma_cond=0,
                 rho_Na=60,
                 rho_K=18,
                 d=0.0476,
                 R_i=0.0345,
                 L=6,
                 N=600,
                 extension=0,
                 dt=0.01,
                 M=1):
        """
        Initialize the neuron. The standard parameters are the original ones,
        see Hodgkin & Huxley (1952), Mathematical Foundations of Neuroscience, 
        together with the approx. equilibrium values as initial conditions.
        Note that the voltage values are shifted by 65mV.
        """
        # Set grid approximation constants
        self.L = L
        self.L_axon = L
        self.N = N
        self.N_axon = N
        self.dx = 1. * L / N
        if extension:
            self.L += extension
            self.N += int(extension / self.dx)
            print 'with extension'
        self.axis_vector = np.linspace(0, self.L, self.N+1, endpoint=True)
        # Number of realizations
        self.M = M
        # Time discretization
        self.dt = dt
        # Initialize state and time
        self.state = np.asarray([init_state[0] * np.ones([self.N+1, M]),
                                 init_state[1] * np.ones([self.N+1, M]),
                                 init_state[2] * np.ones([self.N+1, M]),
                                 init_state[3] * np.ones([self.N+1, M])],
                                dtype='float')
        self.state_grad = np.asarray([np.zeros([self.N-1, M]),
                np.zeros([self.N-1, M]),
                np.zeros([self.N-1, M]),
                np.zeros([self.N-1, M])],
                dtype='float')
        self.time_elapsed = 0
        """
        Coefficients for the deterministic PDE part.
        """
        # Set drift for v
        self.fv = lambda v, m, h, n: f_leak(v) + f_Na(v, m, h) + f_K(v, n)
        # Set inverse time constants for gating variables
        self.t_m = lambda v: alpha_m(v) + beta_m(v)
        self.t_h = lambda v: alpha_h(v) + beta_h(v)
        self.t_n = lambda v: alpha_n(v) + beta_n(v)
        # Set eq values for gating variables
        self.m_inf = lambda v: alpha_m(v) / (alpha_m(v)+beta_m(v))
        self.h_inf = lambda v: alpha_h(v) / (alpha_h(v)+beta_h(v))
        self.n_inf = lambda v: alpha_n(v) / (alpha_n(v)+beta_n(v))
        # Save transition rates
        self.alpha_m = alpha_m
        self.beta_m = beta_m
        self.alpha_h = alpha_h
        self.beta_h = beta_h
        self.alpha_n = alpha_n
        self.beta_n = beta_n
        """
        Noise intensities for current, subunit and conductance noise.
        """
        # Current noise
        self.sigma_curr = sigma_curr * np.sqrt(dt / self.dx)
        # Number of ion channels in each compartment
        N_Na = rho_Na * self.dx * np.pi * d * 10**8
        N_K = rho_K * self.dx * np.pi * d * 10**8
        # Subunit noise
        self.sigma_sub = (sigma_sub * np.sqrt(dt / self.dx) /
                np.asarray([N_Na, N_Na, N_K]))
        # Conductance noise
        self.sigma_cond = sigma_cond
        # Inverse time constants for hidden OU processes.
        self.t_cond = lambda v: np.asarray(
                [self.t_h(v), self.t_m(v), self.t_m(v) * 2, self.t_m(v) * 3,
                 self.t_m(v) * self.t_h(v) * (1/self.t_m(v) + 1/self.t_h(v)),
                 self.t_m(v) * self.t_h(v) * (1/self.t_m(v) + 2/self.t_h(v)),
                 self.t_m(v) * self.t_h(v) * (1/self.t_m(v) + 3/self.t_h(v)),
                 self.t_n(v), self.t_n(v) * 2, self.t_n(v) * 3, self.t_n(v)* 4])
        # Sigma for the OU processes.
        self.sig_cond = lambda v: np.sqrt(np.asarray(
                [self.m_inf(v)**6 * self.h_inf(v) * (1-self.h_inf(v)) / N_Na,
                 3 * self.m_inf(v)**5 * self.h_inf(v)**2 *
                 (1-self.m_inf(v)) / N_Na,
                 3 * self.m_inf(v)**4 * self.h_inf(v)**2 *
                 (1-self.m_inf(v))**2 / N_Na,
                 self.m_inf(v)**3 * self.h_inf(v)**2 *
                 (1-self.m_inf(v))**3 / N_Na,
                 3 * self.m_inf(v)**5 * self.h_inf(v) *
                 (1-self.m_inf(v)) * (1-self.h_inf(v)) / N_Na,
                 3 * self.m_inf(v)**4 * self.h_inf(v) *
                 (1-self.m_inf(v))**2 * (1-self.h_inf(v)) / N_Na,
                 self.m_inf(v)**3 * self.h_inf(v) *
                 (1-self.m_inf(v))**3 * (1-self.h_inf(v)) / N_Na,
                 4 * self.n_inf(v)**7 * (1-self.n_inf(v)) / N_K,
                 6 * self.n_inf(v)**6 * (1-self.n_inf(v))**2 / N_K,
                 4 * self.n_inf(v)**5 * (1-self.n_inf(v))**3 / N_K,
                 self.n_inf(v)**4 * (1-self.n_inf(v))**4 / N_K])) 
        # Initialize the OU processes.
        self.cond_state = np.zeros([11, self.N_axon+1, M])
        """
        Discretization of Laplacian.
        """
        # Matrix entries of discrete Neumann Laplacian A
        A = np.vstack((np.ones(self.N+1),
                       np.hstack((-1, -2 * np.ones(self.N-1), -1))))
        # Multiply with negative diffusion coefficient
        A *= - dt / self.dx**2 * d / (4*R_i)
        # Id - A used in semi-implicit Euler scheme.
        # First and last row already scaled by 0.5 to make A symmetric
        A[1] += 1
        A[1,0] -= 0.5
        A[1,-1] -= 0.5
        self.bandedmatrix = A
        # Constant for input signal through boundary
        self.boundary = 2 * dt / (self.dx * np.pi * d)
        self.output = 0
        # Matrix entries of discrete Dirichlet Laplacian A
        B = np.vstack((np.ones(self.N-1),
                       -2 * np.ones(self.N-1)))
        # Multiply with negative diffusion coefficient
        B *= - dt / self.dx**2 * d / (4*R_i)
        # Id - A used in semi-implicit Euler scheme.
        B[1] += 1
        self.dirichlet = B
        self.dirichletboundary = dt / (np.pi * d)

    def timestep(self, input_signal, output_condition=0,
                 curr_noise=0, sub_noise=0, cond_noise=0):
        """
        Compute the time evolution for one timestep using increments of
        the driving noise (current, subunit or conductance or a mixture)
        and the corresponding input signal at the soma. The output as an
        outflux current can also be specified.
        
        Note that the noise has to be initialized outside the class in
        order to be able to compare models for given trajectories of the
        noise. Shape of the noise has to be
        curr_noise.shape = (N_axon+1, M)
        sub_noise.shape = (3, N_axon+1, M)
        cond_noise.shape = (11, N_axon+1, M)
        """
        [v, m, h, n] = self.state.copy()
        # Explicit Euler(-Maruyama) step for gating variables
        self.state[1] += self.dt * self.t_m(v) * (self.m_inf(v)-m)
        self.state[2] += self.dt * self.t_h(v) * (self.h_inf(v)-h)
        self.state[3] += self.dt * self.t_n(v) * (self.n_inf(v)-n)
        # If subunit noise is switched on
        if np.max(self.sigma_sub):
            N = self.N_axon + 1
            self.state[1][:N, :] += (np.sqrt(self.alpha_m(v[:N, :]) *
                    (1-m[:N, :]) + self.beta_m(v[:N, :]) * m[:N, :]) *
                    self.sigma_sub[0] * sub_noise[0])
            self.state[2][:N, :] += (np.sqrt(self.alpha_h(v[:N, :]) *
                    (1-h[:N, :]) + self.beta_h(v[:N, :]) * h[:N, :]) *
                    self.sigma_sub[1] * sub_noise[1])
            self.state[3][:N, :] += (np.sqrt(self.alpha_n(v[:N, :]) *
                    (1-n[:N, :]) + self.beta_n(v[:N, :]) * n[:N, :]) *
                    self.sigma_sub[2] * sub_noise[2])
        # Respect boundaries!
        for i in xrange(3):
            self.state[i][self.state[i]>1] = 1
            self.state[i][self.state[i]<0] = 0
        # Semi-implicit Euler step for voltage variable
        rhs = v - self.dt*self.fv(v, m, h, n)
        rhs[:self.N_axon+1, :] += self.sigma_curr*curr_noise
        # If conductance noise is switched on
        if self.sigma_cond:
            rhs[:self.N_axon+1,:] -= self.compute_cond_noise(cond_noise)
        # Input signal as Neumann boundary condition for left axon endpoint
        rhs[0] += self.boundary * input_signal
        # Output condition as Neumann boundary condition for right axon endpoint
        rhs[-1] += self.boundary * output_condition
        # Modification of RHS to make discrete Neumann Laplacian symmetric
        rhs[0] *= 0.5
        rhs[-1] *= 0.5
        # Solve the linear problem
        self.state[0] = lng.solveh_banded(self.bandedmatrix, rhs,
                                          check_finite=False)
        # Update state
        #self.state = [v_neu, m_neu, h_neu, n_neu]
        self.time_elapsed += self.dt
    
    def compute_cond_noise(self, bm_inc):
        """
        Compute the conductance noise term driven by the brownian increment
        bm_inc. Returns the effective current related to this noise for easy
        computation in the timestep routine.
        """
        eta = self.cond_state
        v = self.state[0][:self.N_axon+1,:]
        eta = (np.exp(-self.dt*self.t_cond(v))*eta + bm_inc * self.sig_cond(v) *
               np.sqrt(1 - np.exp(-2*self.dt*self.t_cond(v))))
        return (np.sum(eta[0:7], axis=0) * self.fv(v, 1, 1, 0) +
                np.sum(eta[7:-1], axis=0) * self.fv(v, 0, 0, 1))

    def area(self, which=0):
        """
        Compute the area below the traveling pulse.
        Note that this is not shifted by the eq. potential.
        """
        area = (self.dx * np.sum(self.state[which][1:self.N_axon,:], axis=0)
                + self.dx * 0.5 * (self.state[which][0,:]
                + self.state[which][self.N_axon,:]))
        return area
    
    def area_pulse(self, threshold=30, which=0):
        """
        Compute the area below the traveling pulse restricted to
        the voltage being greater than threshold.
        Note that this is not shifted by the eq. potential.
        """
        area = self.dx * np.sum(self.state[which]*
                                (self.state[which] > threshold), axis=0)
        return area

    def area_rest(self, threshold=30, which=0):
        """
        Compute the area below the traveling pulse restricted to
        the voltage being less than threshold.
        Note that this is not shifted by the eq. potential.
        """
        area = (self.dx * np.sum((self.state[which] - threshold)
                *(self.state[which] <= threshold), axis=0)
                + self.L * threshold)
        return area
    
    def area_ext(self, which=0):
        """
        Compute the area below the traveling pulse on the noiseless
        part of the cable.
        """
        if self.N > self.N_axon:
            area = (self.dx * np.sum(self.state[which][self.N_axon+1:-1,:], axis=0)
                    + self.dx * 0.5 * (self.state[which][self.N_axon,:]
                    + self.state[which][-1,:]))
        else:
            area = 0
        return area
        
    def compute_outflux(self):
        """
        Compute the output of the neuron as the outflux in order to
        determine if the pulse has reached the boundary.
        """
        outflux = (self.state[0][-2,:] -
                   self.state[0][-1,:]) / self.boundary
        self.output += outflux
        return self.output
    
    def compute_gradient(self, which=0):
        v = self.state[which].copy()
        return (v[2:] - v[:-2]) / (2*self.dx)        
        
        
        
        
        
    