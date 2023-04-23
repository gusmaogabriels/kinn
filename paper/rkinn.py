#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# Created on Wed Jul 2020
#
# @author:gusmaogabriels@gmail.com // gusmaogabriels@gatech.edu
#
# ---

# + active=""
# !pip install --upgrade jax jaxlib --force
# -

import jax

import jaxlib

jaxlib.__version__

jax.__version__

# %load_ext autoreload
# %autoreload 2

# ### Imports

# + active=""
# -*- coding: utf-8 -*-
# -

#from kude.basis.trainer_ import Trainer, TrainerSeparate, TrainerCV
#from kinn.basis.trainer_uq_inf_rn import TrainerCV
from kinn.basis.model import model
from kinn.basis.nnx import nn
from kinn.basis import plt, jit, optimizers, onp, jnp
from kinn.graphics.generators import gen_gif, gen_sample_gif
from kinn.basis import jnp, random, itertools, clear_output, grad, jacfwd, hessian, vmap
from kinn.basis import plt, MaxNLocator, FormatStrFormatter, HTML, animation, cm,\
                    left, right, bottom, top, wspace, hspace, jnp, make_axes_locatable, griddata
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# Useful activation functions

swish      = jit(lambda x : x/(1.+jnp.exp(-x)))
relu       = jit(lambda x : x*(x>0))
#tanh       = jit(lambda x : jnp.tanh(x))
#sigmoid    = jit(lambda x : jnp.reciprocal(1.+jnp.exp(-x)))
from jax.nn import sigmoid
from jax.nn import swish
from jax.nn import tanh
gauss      = jit(lambda x : jnp.exp(-x**2))
sin        = jit(lambda x : jnp.sin(x))
cos        = jit(lambda x : jnp.cos(x))

# Graphics Aesthetics

# +
import matplotlib as mpl

mpl.rcParams.update({'font.family':'serif'})
mpl.rcParams.update({'font.serif':'Roboto'})
mpl.rcParams.update({'font.sans-serif':'Roboto'})
#mpl.rcParams.update({'font.sans-serif':'FreeSerif'})
mpl.rcParams.update({'mathtext.fontset':'cm'})
#mpl.rcParams.update({'mathtext.fallback_to_cm':True})
mpl.rcParams.update({'font.size':12})
mpl.rcParams.update({'axes.unicode_minus':False})
mpl.rcParams.update({'text.usetex':False})
mpl.rcParams.update({'legend.fontsize': 13.,
          'legend.handlelength': 1})
fs=14.

mpl.rcParams.update({   'figure.titlesize' : fs,
                        })
mpl.rcParams.update({   'axes.titlesize' : fs*11/12.,
                        'axes.labelsize' : fs*10/12.,
                        'lines.linewidth' : 1,
                        'lines.markersize' : fs*10/12.,
                        'xtick.labelsize' : fs*10/12.,
                        'ytick.labelsize' : fs*10/12.})
mpl.rcParams.update({'legend.handletextpad':.4,
                     'legend.handlelength':.6,
                      'legend.columnspacing':.5,
                      'legend.borderaxespad':.5,
                      'legend.fontsize' : fs*10/12.})
mpl.rcParams['axes.linewidth'] = 0.75 #set the value globally
# -

# ## Kinetics Universal Differential Equations (kUDE)
# ### Examples

# ---
#
# ### Paper Examples
#
# **0. Kinetics Framework**  
# **1. Overall Kinetics Definitions** (all types, stoichiometry matrix, parameters)   
#
#     i.   Simple Kinetics Type g (Homogeneous) 
#     ii.  Latent Kinetics Type gda (Heterogeneous) 
#     iii. Latent Kinetics Type gdac (Heterogeneous)
#     iv.  Latent Kinetics Type gdacs** (Heterogeneous)
#

# ## 0. Kinetics Framework

# MF MKM's consist of a set of elementary or irreducible reaction, also referred to as those comprising rare events. Such events involve the interaction of at most two individual entities whose probability of occurrence is proportional to their concentrations, which gives rise to the power law kinetics. In a nut shell, power-law kinetics can be summarized as a linear combination of rare events to occur whose sampling frequency scales linearly with the concentration of each participant, and whose probability of success is given by the Arrhenius law. MKM models comprise the scenario in which one or more of such elementary rare events occur. The mathematical representation of such intertwined set of events embodies a stack of ODE's, which assigns to the rate of change of each state a linear combination of power-law kinetics according to the system's stoichiometries. Let $\mathbf{c}$ represent an array of concentrations or fractions of the system constituents at some given time $t$, and $\mathbf{\dot{c}}$ the rate of change of $\mathbf{c}$, the MKM can be conveyed as in Eq. \ref{eq:kin_ode}.
#
# \begin{equation}
#     \mathbf{\dot{c}}=\mathbf{M}\left(\mathbf{k}(\theta)\odot f(\mathbf{c})\right)\label{eq:kin_ode}\tag{1}
# \end{equation}
#
# Where $\mathbf{M}\in\mathbb{Z}^{n\times m}$ is the corresponding stochiometry matrix and $~{f(\cdot):\mathbb{R}^n_+\to\mathbb{R}^m_+}$ maps $~{\mathbf{c}:=\{\mathbf{c}\,|\,\mathbf{c}\in\mathbb{R}^n_+\}}$ concentrations into the concentration-related parcel of the power-law kinetics and $~{\mathbf{k}:=\{\mathbf{k}(\theta)~\in~\mathbb{R}^m_+,\,\theta\in\mathbb{R}_{+}\}}$ is the temperature- and binding-energies-dependent Arrhenius-like rate constant term. $\mathbf{c}$ encompasses both unbounded, e.g. gas, and bounded (adsorbates or intermediates) species concentration-related state variables, i.e. partial pressures, concentrations, coverage fractions. For further reference, let $\mathbf{c_g} := \{{c}_i\,|\, i\in\mathcal{C}_g\}$ be the subset of $\mathbf{c}$ corresponding to gas-phase species and $\mathbf{c_a}:=\{c_i\,|\,i\in\mathcal{C}_a\}$ is the subset of $\mathbf{c}$ related to bounded gas-counterpart molecules coverage fractions, and $\mathbf{c_s}:=\{c_i\,|\,i\in\mathcal{C}_s\}$ be the subset of $\mathbf{c}$ related to intermediates/radicals species on the catalyst surface, such that $\cup_{i\in\{\mathbf{g},\mathbf{a},\mathbf{s}\}}\mathcal{C}_i=\{1,2,...,m\,|\,m\in\mathbb{N}\}$ and $\cap_{i\in\{\mathbf{g},\mathbf{a},\mathbf{s}\}}\mathcal{C}_i=\emptyset$.
#
# The state-dependent reaction rate vector, denoted as $\mathbf{r}(\mathbf{c},\theta):=\{\mathbf{r}\in\mathbb{R}^m\,|\,\mathbf{r}={k(\theta)}\odot f(\mathbf{c})\}$, comprises a the rate of reaction (frequency) associated with different kinds of elementary reaction: (i) reaction in homogeneous phase, $\mathbf{r_g} := \{{r}_i\,|\, i\in\mathcal{G}_g\}$, (ii) those involving adsorption-desorption, $\mathbf{r_d} := \{{r}_i\,|\, i\in\mathcal{G}_d\}$, , (iii) reactions between adsorbed molecules, $\mathbf{r_a} := \{{r}_i\,|\, i\in\mathcal{G}_a\}$, (iv) reactions involving adsorbed molecules and radicals/intermediates on the surface, $\mathbf{r_c} := \{{r}_i\,|\, i\in\mathcal{G}_c\}$, and (v) reactions between radicals on the surface, $\mathbf{r_s} := \{{r}_i\,|\, i\in\mathcal{G}_s\}$, such that $\cup_{i\in\{\mathbf{g},\mathbf{d},\mathbf{a},\mathbf{c},\mathbf{s}\}}\mathcal{G}_i=\{1,2,...,n\,|\,n\in\mathbb{N}\}$ and $\cap_{i\in\{\mathbf{g},\mathbf{d},\mathbf{a},\mathbf{c},\mathbf{s}\}}\mathcal{G}_i=\emptyset$. The stoichiometry matrix in turn consits of a composition of reaction-types and submatrices constituents of the full stoichiometry matrix, as follows in Eq. \ref{eq:stoich_full}.
#
# \begin{align}
#     \begin{bmatrix} \mathbf{\dot{c}_g}\\\mathbf{\dot{c}_a}\\\mathbf{\dot{c}_s}\end{bmatrix}=\begin{bmatrix} \mathbf{M_{gg}}&\mathbf{{M_{gd}}}&\mathbf{0}&\mathbf{0}&\mathbf{0}\\\mathbf{0}&\mathbf{{M_{ad}}}&\mathbf{{M_{aa}}}&\mathbf{{M_{ac}}}&\mathbf{0}\\\mathbf{0}&\mathbf{0}&\mathbf{0}&\mathbf{M_{sc}}&\mathbf{M_{ss}}\end{bmatrix}\mathbf{r}(\mathbf{c},\theta)
#     \label{eq:stoich_full}\tag{2}
# \end{align}
#
# The full stoichiometry matrix in Eq. \ref{eq:stoich_full} generally conveys the different types of chemical kinetics: (i) purely homogeneous reactions, only $\mathbf{M_{:g}}\ne\mathbf{0}$, (ii) pure adsorption/desorption reactions, $\mathbf{M_{:\{gd\}}}\ne\mathbf{0}$, (iii) single intermediate surface reactions, $\mathbf{M_{:\{gdac\}}}\ne\mathbf{0}$, and (iv) reaction containing elementary reactions involving surface intermediates, $\mathbf{M_{:\{gdacs\}}}\ne\mathbf{0}$, where $:$ denotes all rows and $\{\mathbf{i}\}\subset\{\mathbf{g},\mathbf{d},\mathbf{a},\mathbf{c},\mathbf{s}\}$ are the corresponding reaction types $\mathbf{i}$ as columns of $\mathbf{M}$. This framework allows for further classification of types of kinetics solely based on the chemical reaction stoichiometry matrix.

# #### BC-Constrained Neural Net
#
# What we do here is to use an operator that enforces $\mathbf{x}(\omega_{sm},t_0)=\mathbf{x}_0~\forall~\omega_{sm}$. A natural choice for this purpose is $c(t)=tanh(t)\in \mathcal{C}^1$, since $tanh(0)=0$ and $\partial_t\tanh(0)=1$. Let $\mathbf{x}(\omega_{sm},t)$ denote a surrogate model, in this case a neural network with parameters $\omega_{sm}$ and independent variable $t$, the boundary condition operator is defined as $\mathbf{C}[\mathbf{x}(\omega_{sm},t),\mathbf{x}_0]=\mathbf{x}(\omega_{sm},t)c(t-t_0)+\mathbf{x}_0$. 
#
# From simple inspection, we have that 
# $$\mathbf{x}(\omega_{sm},t)c(t-t_0)|_{t=t_0}+\mathbf{x}_0=\mathbf{x}(\omega_{sm},t_0)c(0)+\mathbf{x}_0=\mathbf{x}_0$$
# $$\partial_t~\mathbf{x}(\omega_{sm},t)c(t-t_0)|_{t=t_0}+\mathbf{x}_0=\mathbf{x}_t(\omega_{sm},t_0)c(0)+\mathbf{x}(\omega_{sm},t_0)c_t(0)=\mathbf{x}(\omega_{sm},t_0)$$
#
# Such that the neural net vanishes at $t=t_0$ with continuous time-derivative equals the neural net output. Therefore, any *Dirichlet* boundary condition can be automatically satisfied in this form for UDEs. The same approach can be extend to *Neumann* boundary conditions with proper choice of $c$ function.
#
# $$\mathbf{\dot{x}}(\omega_{sm},t)=Mf(\mathbf{x}(\omega_{sm},t))$$
# $$\mathbf{\dot{x}}(\omega_{sm},t_0)=Mf(\mathbf{x}(\omega_{sm},t_0))$$
# $$\mathbf{x}(\omega_{sm},t)=\mathbf{C}[\mathbf{x'}(\omega_{sm},t)]=\mathbf{x'}(\omega_{sm},t)c(t)$$
# $$\partial_t[{\mathbf{x'}(\omega_{sm},t)c(t-t_0)}+\mathbf{x}_0]=Mf(\mathbf{x'}(\omega_{sm},t)c(t-t_0)+\mathbf{x}_0)$$
# $$\mathbf{\dot{x}'}(\omega_{sm},t)c(t-t_0)+\mathbf{x'}(\omega_{sm},t)\dot{c}(t-t_0)=Mf(\mathbf{x'}(\omega_{sm},t)c(t-t_0)+\mathbf{x}_0)$$
# $$\mathbf{x'}(\omega_{sm},t_0)=Mf(\mathbf{x}_0)$$
#
# Below `nn_c` is recasted from the parent `nn` function by adding $\mathcal{C}[x]$ as a constraint.

# ### Hypersphere Contrained Subclassed NN for Latent Variables
#
# #### General Definition
#
# Kinetic models under mean-field approximation deal with states as the evolution of descriptive statistics over the scale under considerations. In particular, MKM treat concentration of intermediate species in terms of fractions of the total number of active sites available. As a result, it is desirable to develop surrogate models that structurally enforce normalization by eliminating the additional degree of freedom. To this end, we propose the use of the projection od radius-one hypersphere onto the natural basis. 
#
# Let $\mathbf{x_s}(\omega_{sm},t)\in\mathbb{R}^p$ be the output values of the constrained surrogate related to the neural network $x'(\omega_{sm},t)\in\mathbb{R}^{p-1}$, such that
#
# \begin{align}
#     {x_s}_i(\omega_{sm},t)&=\left(1-\sin^{2}\left({x'}_i(\omega_{sm},t)\right)\right)\prod_{j<i}\sin^{2}\left({x'}_j(\omega_{sm},t)\right)\;\forall\;i<p;\;{i,j}\in\mathbb{N}\\
#     {x_s}_p(\omega_{sm},t)&=\prod_{j<p}\sin^{2}\left({x'}_j(\omega_{sm},t)\right)
# \end{align}
#
# Such trigonometric transformation enforces that $0\le\mathbf{x_s}\le1$ and $\sum\mathbf{x_s}=1\;\forall\;\mathbf{x'}\in\mathbb{R}^{p-1}$.

# #### Hypersphere Constrained with Boundary Conditions 
# *Dirichlet*
#
# Since the hypersphere constraint maps from $\mathbb{R}^{p-1}$ to $\mathbb{R}^p$, applying boundary conditions operators on the surrogate model output would lead to increasing stiffness in the underlying neural net training (**hand-wavy: may need additional math elaboration**). The workaround is to transform the output layer boundary conditions to the output of the inner neural network, i.e. convert the boundary contidition to equivalent hypersphere angles. The latter can be undertaken by properly finding the bijection from the hypersphere transformation, as follows.
#
# \begin{align}
#     {x'_0}_i=\arcsin{\left(\sqrt{1-\frac{{x_0}_i}{\prod_{j<i}\sin^2\left({x_0}_j\right)}}\right)};\forall\;i<p;\;i,j\in\mathbb{N}
# \end{align}
#
# Where $\mathbf{x'_0}$ is the angle-transformed boundary conditions and $\mathbf{x_0}$ is the natural boundary condition for the surrogate model output. For the degenerate case where one of the outputs is the unit, the corresponding angle should be set to $0$ and all other angles to $\frac{\pi}{2}$.

# ## 1. Overall Kinetics Definitions

# ### i. Simple Kinetics Type gg

# The simple initial model represents the following fully-reversible chemical reaction:
#
# $$A+B\underset{k_2}{\stackrel{k_1}{\rightleftharpoons}} C\notag$$

# ### ii. Latent Kinetics Type gda

# The Latent Kinetics type gad involves ad/desorption steps and a surface reaction between adsorbed molecules.
#
# $$A\underset{k_2}{\stackrel{k_1}{\rightleftharpoons}} A*\notag\\
# B\underset{k_4}{\stackrel{k_3}{\rightleftharpoons}} B*\notag\\
# C\underset{k_6}{\stackrel{k_5}{\rightleftharpoons}} C*\notag\\
# A*+B*\underset{k_8}{\stackrel{k_7}{\rightleftharpoons}} C*\notag$$

# ### iii. Latent Kinetics Type gdac

# An intermediate species (radicals, $D*$) that do not have a corresponding gas phase species is part of the reaction.
#
# $$A\underset{k_2}{\stackrel{k_1}{\rightleftharpoons}} A*\notag\\
# B\underset{k_4}{\stackrel{k_3}{\rightleftharpoons}} B*\notag\\
# C\underset{k_6}{\stackrel{k_5}{\rightleftharpoons}} C*\notag\\
# B*+*\underset{k_8}{\stackrel{k_7}{\rightleftharpoons}} 2D*\notag\\
# A*\:+\:D*\underset{k_{10}}{\stackrel{k_9}{\rightleftharpoons}} C*\:+\:*\notag$$

# ### iv. Latent Kinetics Type gdacs

# Reaction between radicals $D*$, $E*$ and $F*$ adds further complexity to surface reaction.
#
# $$A\underset{k_{-1}}{\stackrel{k_1}{\rightleftharpoons}} A*\notag\\
# B\underset{k_{-2}}{\stackrel{k_2}{\rightleftharpoons}} B*\notag\\
# C\underset{k_{-3}}{\stackrel{k_3}{\rightleftharpoons}} C*\notag\\
# A*+*\underset{k_{-4}}{\stackrel{k_4}{\rightleftharpoons}} 2D*\notag\\
# B*+*\underset{k_{-5}}{\stackrel{k_5}{\rightleftharpoons}} 2E*\notag\\
# D*\:+\:E*\underset{k_{-6}}{\stackrel{k_{6}}{\rightleftharpoons}} F*\:+\:*\notag\\
# F*\:+\:E*\underset{k_{-7}}{\stackrel{k_{13}}{\rightleftharpoons}} C*\:+\:*\notag$$

# Concentrations of chemical species $A$, $B$ and $C$ are observable variables, i.e. for topics [1], [2] and [3], there are not any latent (hidden) variables.

# A dictionary of stoichiometry matrices and with kinetic parameters (rate constants).
#
# All reactions are considered reversible

# +
pars = {i:dict() for i in range(5)}

# visible boundary conditions
t0                  = 0.
#bcv                 = [[0.6,0.4,0.0],[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]]
#bcv                 = [[0.6,0.4,0.0],[0.2,0.3,0.5],[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]]
#bcv                 = [[0.6,0.4,0.0]]
#bcv = [[0.2,0.3,0.5]]
#bcv = [[0.9,0.05,0.05]]
#bcv                 = [[0.05,0.05,0.9],[0.6,0.4,0.0]]

bcv                 = [[0.2,0.3,0.5],[0.6,0.4,0.0]]
#bcv                 = [[0.6,0.4,0.0],[0.2,0.3,0.5]]
#bcv                 = [[0.6,0.4,0.0]] 
trig                = True

pars[0]['type']     = 'homogeneous'
pars[0]['sps']      = ['A','B','C']
pars[0]['stoich']   = jnp.array([[-1,-1, 1],
                                [ 1, 1,-1]]).T
pars[0]['kinpars']  = jnp.array([10.,1.])
pars[0]['bc']       = [[jnp.array(_) for _ in [[t0],bcv_]] for bcv_ in bcv]
pars[0]['nncpars']  = [{'forward': {'predictor':{'trig':False, 'damp':True,'nobs':3,'bc':bc_,'nn_scale': 0.001}},\
                       'inverse': {'predictor':{'trig':False, 'damp':False,'nobs':3,'bc':None,'nn_scale': 0.001},\
                                    'inference':{'trig':False, 'damp':False,'nobs':0,'bc':None,'nn_scale': 0.001}}} for bc_ in pars[0]['bc']]
pars[0]['nnmpars']  = {'forward': {'layers_sizes'  : [[1,5,5,5,pars[0]['stoich'].shape[0]]],
                       'act_fun'                   : [sigmoid,swish,sigmoid],
                       'nn_scale'                  : 0.001},
                       'inverse' : {'layers_sizes' : [[1,5,5,pars[0]['stoich'].shape[0]]],
                       'act_fun'                   : [swish,sigmoid],
                       'nn_scale'                  : 0.001}}
pars[0]['nntpars']  = {'layers_sizes' : [[1,3,1]],
                       'act_fun'      : [swish,tanh,swish,tanh],
                       'nn_scale'     : 0.001}

pars[1]['type']     = 'heterogeneous'
pars[1]['sps']      = ['A','B','C','A*','B*','C*','*']
pars[1]['stoich']   = jnp.array([[-1, 0, 0, 1, 0, 0,-1],
                                [ 1, 0, 0,-1, 0, 0, 1],
                                [ 0,-1, 0, 0, 1, 0,-1],
                                [ 0, 1, 0, 0,-1, 0, 1],
                                [ 0, 0,-1, 0, 0, 1,-1],
                                [ 0, 0, 1, 0, 0,-1, 1],
                                [ 0, 0, 0,-1,-1, 1, 1],
                                [ 0, 0, 0, 1, 1,-1,-1]]).T
pars[1]['kinpars']  = jnp.array([.5,.2,2.,3.,10.,2.,5.,4])*20.
pars[1]['bc']       = [[jnp.array(_) for _ in [[t0],bcv_+[0.]*(pars[1]['stoich'].shape[0]-1-len(bcv_))+[1.]]] for bcv_ in bcv]
pars[1]['nncpars']  = [{'forward': {'predictor':{'trig':True, 'damp':True,'nobs':3,'bc':bc_,'nn_scale': 0.001}},\
                       'inverse': {'predictor':{'trig':True, 'damp':False,'nobs':3,'bc':None,'nn_scale': 0.001},\
                                    'inference':{'trig':True, 'damp':False,'nobs':0,'bc':None,'nn_scale': 0.001}}} for bc_ in pars[1]['bc']]
pars[1]['nnmpars']  = {'forward': {'layers_sizes'  : [[1,10,10,10,pars[1]['stoich'].shape[0]-1]],
                       'act_fun'                   : [sigmoid,swish,sigmoid],
                       'nn_scale'                  : 0.001},
                       'inverse' : {'layers_sizes' : [[1,15,15,pars[1]['stoich'].shape[0]-1]],
                       'act_fun'                   : [swish,swish],#[swish,jit(lambda x : jnp.exp(-x**2)),swish],#[tanh,swish,tanh,tanh,swish],
                       'nn_scale'                  : 0.001},
                       'inference' : {'layers_sizes' : [[1,10,10,10,pars[1]['stoich'].shape[0]-1-pars[1]['nncpars'][0]['inverse']['predictor']['nobs']]],
                       'act_fun'                   : [swish,swish,tanh],#[swish,jit(lambda x : jnp.exp(-x**2)),swish],#[tanh,swish,tanh,tanh,swish],
                       'nn_scale'                  : 0.001}}
pars[1]['nntpars']  = {'layers_sizes' : [[1,3,1]],
                       'act_fun'      : [swish,tanh,swish,tanh],
                       'nn_scale'     : 0.001}

pars[2]['type']     = 'heterogeneous'
pars[2]['sps']      = ['A','B','C','A*','C*','*']
pars[2]['stoich']   = jnp.array([[-1, 0, 0, 1, 0,-1],
                                [ 1, 0, 0,-1, 0, 1],
                                [ 0, 0,-1, 0, 1,-1],
                                [ 0, 0, 1, 0,-1, 1],
                                [ 0,-1, 0,-1, 1, 0],
                                [ 0, 1, 0, 1,-1, 0]]).T
pars[2]['kinpars']  = jnp.array([.5,.2,1.,2.,5.,4])*20.
pars[2]['bc']       = [[jnp.array(_) for _ in [[t0],bcv_+[0.]*(pars[2]['stoich'].shape[0]-1-len(bcv_))+[1.]]] for bcv_ in bcv]
pars[2]['nncpars']  = [{'forward': {'predictor':{'trig':True, 'damp':True,'nobs':3,'bc':bc_,'nn_scale': 0.001}},\
                       'inverse': {'predictor':{'trig':True, 'damp':False,'nobs':3,'bc':None,'nn_scale': 0.001},\
                                    'inference':{'trig':True, 'damp':False,'nobs':0,'bc':None,'nn_scale': 0.001}}} for bc_ in pars[2]['bc']]
pars[2]['nnmpars']  = {'forward': {'layers_sizes'  : [[1,8,8,8,pars[2]['stoich'].shape[0]-1]],
                       'act_fun'                   : [sigmoid,swish,sigmoid],
                       'nn_scale'                  : 0.001},
                       'inverse' : {'layers_sizes' : [[1,12,12,pars[2]['stoich'].shape[0]-1]],
                       'act_fun'                   : [swish,swish],#[swish,jit(lambda x : jnp.exp(-x**2)),swish],#[tanh,swish,tanh,tanh,swish],
                       'nn_scale'                  : 0.001},
                       'inference' : {'layers_sizes' : [[1,8,8,8,pars[2]['stoich'].shape[0]-1-pars[2]['nncpars'][0]['inverse']['predictor']['nobs']]],
                       'act_fun'                   : [swish,swish,swish],#[swish,jit(lambda x : jnp.exp(-x**2)),swish],#[tanh,swish,tanh,tanh,swish],
                       'nn_scale'                  : 0.001}}
pars[2]['nntpars']  = {'layers_sizes' : [[1,3,1]],
                       'act_fun'      : [swish,tanh,swish,tanh],
                       'nn_scale'     : 0.001}

pars[3]['type']     = 'heterogeneous'
pars[3]['sps']      = ['A','B','C','A*','B*','C*','D*','*']
pars[3]['stoich']   = jnp.array([[-1, 0, 0, 1, 0, 0, 0,-1],
                                [ 1, 0, 0,-1, 0, 0, 0, 1],
                                [ 0,-1, 0, 0, 1, 0, 0,-1],
                                [ 0, 1, 0, 0,-1, 0, 0, 1],
                                [ 0, 0,-1, 0, 0, 1, 0,-1],
                                [ 0, 0, 1, 0, 0,-1, 0, 1],
                                [ 0, 0, 0, 0,-1, 0, 2,-1],
                                [ 0, 0, 0, 0, 1, 0,-2, 1],
                                [ 0, 0, 0,-1, 0, 1,-1, 1],
                                [ 0, 0, 0, 1, 0,-1, 1,-1]]).T
pars[3]['kinpars']  = jnp.array([.5,.2,.4,.1,.3,.2,30.,10.,50.,40.])*40.
pars[3]['bc']       = [[jnp.array(_) for _ in [[t0],bcv_+[0.]*(pars[3]['stoich'].shape[0]-1-len(bcv_))+[1.]]] for bcv_ in bcv]
pars[3]['nncpars']  = [{'forward': {'predictor':{'trig':True, 'damp':True,'nobs':3,'bc':bc_,'nn_scale': 0.001}},\
                       'inverse': {'predictor':{'trig':True, 'damp':False,'nobs':3,'bc':None,'nn_scale': 0.001},\
                                    'inference':{'trig':True, 'damp':False,'nobs':0,'bc':None,'nn_scale': 0.001}}} for bc_ in pars[3]['bc']]
pars[3]['nnmpars']  = {'forward': {'layers_sizes'  : [[1,12,12,12,pars[3]['stoich'].shape[0]-1]],
                       'act_fun'                   : [sigmoid,swish,sigmoid],
                       'nn_scale'                  : 0.001},
                       'inverse' : {'layers_sizes' : [[1,18,pars[3]['stoich'].shape[0]-1]],
                       'act_fun'                   : [swish],
                       'nn_scale'                  : 0.001},
                       'inference' : {'layers_sizes' : [[1,16,16,16,pars[3]['stoich'].shape[0]-1-pars[2]['nncpars'][0]['inverse']['predictor']['nobs']]],
                       'act_fun'                   : [swish,swish,swish],
                       'nn_scale'                  : 0.001}}
pars[3]['nntpars']  = {'layers_sizes' : [[1,6,1]],
                       'act_fun'      : [swish,tanh,swish,tanh],
                       'nn_scale'     : 0.001}

pars[4]['type']     = 'heterogeneous'
pars[4]['sps']      = ['A','B','C','A*','B*','C*','D*','E*','F*','*']
pars[4]['stoich']   = jnp.array([[-1, 0, 0, 1, 0, 0, 0, 0, 0,-1],
                                [ 1, 0, 0,-1, 0, 0, 0, 0, 0, 1],
                                [ 0,-1, 0, 0, 1, 0, 0, 0, 0,-1],
                                [ 0, 1, 0, 0,-1, 0, 0, 0, 0, 1],
                                [ 0, 0,-1, 0, 0, 1, 0, 0, 0,-1],
                                [ 0, 0, 1, 0, 0,-1, 0, 0, 0, 1],
                                [ 0, 0, 0,-1, 0, 0, 2, 0, 0,-1],
                                [ 0, 0, 0, 1, 0, 0,-2, 0, 0, 1],
                                [ 0, 0, 0, 0,-1, 0, 0, 2, 0,-1],
                                [ 0, 0, 0, 0, 1, 0, 0,-2, 0, 1],
                                [ 0, 0, 0, 0, 0, 0,-1,-1, 1, 1],
                                [ 0, 0, 0, 0, 0, 0, 1, 1,-1,-1],
                                [ 0, 0, 0, 0, 0, 1, 0,-1,-1, 1],
                                [ 0, 0, 0, 0, 0,-1, 0, 1, 1,-1]]).T
pars[4]['kinpars']  = jnp.array([25.,10.,30.,15.,20.,50.,800.,1200.,200.,100.,800.,300.,700.,200.])*0.8
pars[4]['bc']       = [[jnp.array(_) for _ in [[t0],bcv_+[0.]*(pars[4]['stoich'].shape[0]-1-len(bcv_))+[1.]]] for bcv_ in bcv]
pars[4]['nncpars']  = [{'forward': {'predictor':{'trig':True, 'damp':True,'nobs':3,'bc':bc_,'nn_scale': 0.001}},\
                        'inverse': {'predictor':{'trig':True, 'damp':False,'nobs':3,'bc':None,'nn_scale': 0.001},\
                                    'inference':{'trig':True, 'damp':False,'nobs':0,'bc':None,'nn_scale': 0.001}}} for bc_ in pars[4]['bc']]
pars[4]['nnmpars']  = {'forward': {'layers_sizes'  : [[1,5,5,5,pars[4]['stoich'].shape[0]-1]],
                       'act_fun'                   : [sigmoid,swish,sigmoid],
                       'nn_scale'                  : 0.001},
                       'inverse' : {'layers_sizes' : [[1,20,20,20,pars[4]['stoich'].shape[0]-1]],
                       'act_fun'                   : [swish]*3,
                       'nn_scale'                  : 0.001},
                      'inference' : {'layers_sizes' : [[1,20,20,20,pars[4]['stoich'].shape[0]-1-pars[3]['nncpars'][0]['inverse']['predictor']['nobs']]],
                       'act_fun'                   : [swish,swish,swish],
                       'nn_scale'                  : 0.001}}
pars[4]['nntpars']  = {'layers_sizes' : [[1,6,1]],
                       'act_fun'      : [swish,tanh,swish],
                       'nn_scale'     : 0.001}
# -

# ### a. Forward Problems
#
# $bc_0=[0.5,0.5,0.]$ for all models
#
# Inspired by [**Solving coupled ODEs with a neural network and autograd**](http://kitchingroup.cheme.cmu.edu/blog/category/ode/ ) (Kitchin's Group)

# Model parameters (rate constants): `model_params0`  
# Boundary conditions (concentrations at $t=0$): `bc0`

# Model function, `x` are state variables, `t` the independet variable (e.g. time).

# from trainer_source_uq_sep import *
# from trainer_source_uq_sep import trainer_ as trainer_
from trainer_source import *
from trainer_source import trainer as trainer_

# +
__reload__ = False
__errors__ = {}
__results__ = {}

addon = '_alpha107sens_uq_full'
#_alphas = jnp.concatenate((jnp.logspace(-6,6,13),jnp.logspace(-4,8,13)[::-1]))
#_alphas = jnp.concatenate((jnp.logspace(-6,6,7),jnp.logspace(-6,6,7)[-2::-1]))
_alphas = jnp.concatenate((jnp.logspace(-6,6,21),jnp.logspace(-6,6,41)[-2::-1]))
cutoff      = 1e-1#-1
#for no in range(4):
#for no in range(1,4):

__pars_results__ = dict()

from jax.config import config
config.update('jax_disable_jit', False)
#for no in range(1,5):
for no in [4]:
    __pars_results__[no] = dict()
    #for tag, mode in zip(['fwd','inv','invsc','invvwn'],['forward']+['inverse']*3):
    for tag, mode in zip(['invvwn'],['inverse']):
        
        if tag.startswith('inv'):
            alphas = _alphas
        else:
            alphas = [1]

        model_scale = 1e-3 # model scale (assume low so as not to be bias)
        model_nn = model(pars[no]['stoich'],model_scale=model_scale)
        #model_nn.U = jnp.eye(model_nn.U.shape[0])
        #model_nn.params = [pars[no]['kinpars']]
        if tag == 'fwd':
            model_nn.params = [pars[no]['kinpars']]

        model_ = model(pars[no]['stoich'])
        model_.params = [pars[no]['kinpars']]

        def gen_data(n_points,i):
            @jit
            def ode(t,C):
                return model_.single_eval([pars[no]['kinpars']],[t,C]).flatten()

            tmax = 20 # max time to evaluate
            #t_eval = jnp.logspace(0,jnp.log10(tmax),n_points)-1.
            t_eval = (jnp.logspace(0,jnp.log10(tmax+1),n_points)-1.)/tmax
            #t_eval = jnp.linspace(0,1,n_points)
            print((pars[no]['bc'][i][0], tmax), pars[no]['bc'][i][1])
            sol = solve_ivp(ode, (pars[no]['bc'][i][0], tmax), pars[no]['bc'][i][1], t_eval = t_eval, method='LSODA',atol=1e-20,rtol=1e-20)

            return t_eval.reshape(-1,1), sol.y.T

        nnms = []
        #nnin = []
        nnts = []
        nncs = [] # one per dataset
        nnis = [] # for inference
        for i in range(len(pars[no]['bc'])):
            nnm = nn(**pars[no]['nnmpars'][mode])
            nnms += [nnm]
            if mode == 'forward':
                inference = False
                nnt = nn(**pars[no]['nntpars'])
                nnts += [nnt]
                nncs += [nn_npt([nnm,nnt], mode=mode,**pars[no]['nncpars'][i][mode]['predictor'],Ur=model_nn.Ur,Un=model_nn.Un)]
            elif mode == 'inverse': 
                if tag == 'inv':
                    inference = True
                    nncs += [nn_npt([nnm], mode=mode,**pars[no]['nncpars'][i][mode]['predictor'],Ur=model_nn.Ur,Un=model_nn.Un)]
                else:
                    inference = False
                    nncs += [nn_npt([nnm], mode=mode,**pars[no]['nncpars'][i][mode]['predictor'],Ur=model_nn.Ur,Un=model_nn.Un)]
            else:
                raise Exception('mode not implemented ({})'.format(mode))
        
        num_epochs = 100
        num_iter   = 300

        if tag.endswith('sc') or tag.endswith('wn'):
            scale = True
        else:
            scale = False

        print('scale',scale)
        tobj = trainer_(nncs, model_nn, num_iter=num_iter, num_epochs=num_epochs, batch_size=1.,\
                               split=1., verbose=True, mode=mode, scale=scale, historian=True, tol=1e-10, stol=0.01, \
                               nobs=pars[no]['nncpars'][i][mode]['predictor']['nobs'],iter_data=[],inference=inference, \
                               kl_inference=False)
        #tobj.__setattr__('err_tags',['MLE MODEL','MLE MODEL (NULL)','MLE DATA','MLE DATA (NULL)','ENTROPY'])
        tobj.__setattr__('err_tags',['MLE MODEL','MLE DATA'])

        if __reload__:# and not (tag.endswith('sc') or tag.endswith('wn')):
            try:
                tobj.load('trainer_{}_{}{}.npz'.format(tag,no,addon))
            except:
                print('Load failed: {} {}'.format(tag,no))
                __errors__.update({'LOAD_FAILED':(tag,no)})


        datas0 = []
        datas0_diff = []
        for i in range(len(pars[no]['bc'])):
            t, x = gen_data(100,i)
            data0 = [(t,x.copy(),[],model_.batched_eval([pars[no]['kinpars']],(t,x)))]
            datas0 += data0
        dstack = jnp.vstack([_[1][:,pars[no]['nncpars'][i][mode]['predictor']['nobs']:] for _ in datas0])
        s_ = jnp.std(dstack,axis=0)
        #for tol, h, n_points in zip([1e-3,1e-4,1e-12],[1e-4,1e-5,1e-6],[40]*3):     

        datas  = []

        for i in range(len(pars[no]['bc'])):
            
#             dstack = jnp.vstack([_[1][:,pars[no]['nncpars'][i][mode]['predictor']['nobs']:] for _ in datas0])
#             s_ = jnp.std(dstack,axis=0)

            t, x, [], dx = datas0[i]
            #t_ = t[:-1,:]+jnp.diff(t,axis=0)/2
            #t_ = jnp.sort(jnp.concatenate([t[:-1,:]+jnp.diff(t,axis=0)/6*(i+1) for i in range(5)]),axis=0)
            t_ = []
            
            if tag == 'fwd': 
                data = [(t,[],t_)]
            elif tag == 'inv':
                xinv  = x[:,:pars[no]['nncpars'][i][mode]['predictor']['nobs']] 
                xinv  = xinv + random.normal(random.PRNGKey(0),xinv.shape)*0.00
                xinv = xinv*(xinv>=0.)
                data = [(t,xinv,t_)]
            elif tag == 'invsc':
                d = x[:,pars[no]['nncpars'][i][mode]['predictor']['nobs']:]
                d = d/s_
                #d = (d-d.mean(axis=0))/d.std(axis=0)
                #d =  d+d.min(axis=0)
                x_sc = jnp.hstack((x[:,:pars[no]['nncpars'][i][mode]['predictor']['nobs']],d))
                data = [(t,x_sc,t_)]
            elif tag == 'invvwn':
                d = x[:,pars[no]['nncpars'][i][mode]['predictor']['nobs']:]
                d = d/s_
                #d = (d-d.mean(axis=0))/d.std(axis=0)
                #d =  d+d.min(axis=0)
                x_sc = jnp.hstack((x[:,:pars[no]['nncpars'][i][mode]['predictor']['nobs']],d))
                x_scwn = jnp.abs(x_sc + random.normal(random.PRNGKey(0),x_sc.shape)*0.025)
#                 x_scwn = x_sc + jnp.hstack((random.normal(random.PRNGKey(0),x_sc[:,:pars[no]['nncpars'][i][mode]['predictor']['nobs']].shape)*0.025,
#                                            random.normal(random.PRNGKey(0),x_sc[:,pars[no]['nncpars'][i][mode]['predictor']['nobs']:].shape)*0.025))#*jnp.sqrt(jnp.reciprocal(s_))))
#                 x_scwn = x_sc + jnp.hstack((random.normal(random.PRNGKey(0),x_sc[:,:pars[no]['nncpars'][i][mode]['predictor']['nobs']].shape)*0.01,
#                                            random.normal(random.PRNGKey(0),x_sc[:,pars[no]['nncpars'][i][mode]['predictor']['nobs']:].shape)*0.005*jnp.reciprocal(jnp.mean(s_))))
                #x_scwn = x_scwn*(x_scwn>=0.)
                data = [(t,x_scwn,t_)]
            datas  += data
        tobj.tol  = 1e-1
        tobj.rtol = 1e-5
        tobj.kltol = 1e-5
        adam_kwargs  = {'step_size':1e-3,'eps':1e-12,'b1':0.9,'b2':0.999}
        tobj.set_optimizer(optimizers.adam,adam_kwargs)
        tobj.refresh = 1e-1
        tobj.rejit()
        for step in range(100):
            print(no,tag,step)
            tobj.train(datas,alpha=1.,isig=1.,beta=.5)
        
        __pars_results__[no][tag] = [pars[no]['kinpars'][:],jnp.exp(tobj.params[1][0])[:]]

        __results__['trainer_{}_{}{}'.format(tag,no,addon)] = tobj.get_error_metrics([_[:3] for _ in datas0])
        tobj.dump('trainer_{}_{}{}'.format(tag,no,addon))
# -
len(datas[0][-1])

[jnp.trace(_) for _ in tobj.sigms[1]]

jnp.mean(jnp.array([jnp.trace(_) for _ in tobj.cov_x]))

e0, P0 = jnp.linalg.eigh(jnp.cov(jnp.vstack(datas0[0][1][:,tobj.nobs:] ).T))
e1, P1 = jnp.linalg.eigh(jnp.cov(jnp.vstack(datas0[1][1][:,tobj.nobs:] ).T))
d = jnp.vstack((datas[_][1][:,tobj.nobs:] ) for _ in range(2))

d0 = datas0[1][1]
d_ = datas[1][1]
d = jnp.hstack((d_[:,:3],d_[:,3:]*tobj.scales))
d2 = jnp.hstack((d_[:,:3],d_[:,3:]*s_))
plt.plot(d.dot(tobj.Un));
#plt.plot(d2.dot(tobj.Un));
plt.plot(d0.dot(tobj.Un))

(d.dot(tobj.Un)).std(axis=0),\
(d2.dot(tobj.Un)).std(axis=0)

ss = jnp.array([0.12822571, 0.02528004, 0.05609103, 0.01540709, 0.05294403, 0.00816492, 0.11483553])
ss = jnp.array([0.00366484, 0.04046658, 0.05814725, 0.01198164, 0.10958327, 0.01607345, 0.0908332])
ss = jnp.array([0.05721629, 0.01395772, 0.03228229, 0.05802602, 0.05151136, 0.02439292, 0.11949582])
ss = jnp.array([0.05931345, 0.01470867, 0.03419821, 0.05830543, 0.05176885, 0.0222256, 0.11886306])

plt.plot(s_,ss,'.')
plt.plot([0,0.15],[0,0.15])

plt.plot(s_,tobj.scales,'.')
# tobj.stol = 1e-2
# plt.plot(s_,tobj.get_scales(datas),'.-')
plt.plot([0,0.15],[0,0.15])

yls = [datas[_][1][:,tobj.nobs:]-0*datas[_][1][:,tobj.nobs:].mean(axis=0) for _ in range(len(datas))]
xos = [datas[_][1][:,:tobj.nobs]-0*datas[_][1][:,:tobj.nobs].mean(axis=0) for _ in range(len(datas))]
d = jnp.vstack(yls)
#e, P = jnp.linalg.eigh(jnp.mean(jnp.array([jnp.cov(yl.T) for yl in yls]),axis=0))
e, P = jnp.linalg.eigh(jnp.cov(d.T))

plt.hist((jnp.array(datas0[0][1].dot(tobj.Un)-jnp.hstack((datas[0][1][:,:3],datas[0][1][:,3:]*tobj.scales)).dot(tobj.Un)).T).tolist(),bins=50,)

#plt.plot(datas[0][1][:,3:]*s_)
#plt.plot(datas[0][1][:,3:]*tobj.scales)
plt.plot(datas[0][1][:,:3])

yl = jnp.vstack((datas[_][1][:,tobj.nobs:] for _ in [1]))#range(len(datas))))
xo = jnp.vstack((datas[_][1][:,:tobj.nobs] for _ in [1]))#range(len(datas))))
e, P = jnp.linalg.eigh(jnp.cov(yl.T))
#e, P = jnp.linalg.eigh(yl.T.dot(yl,precision=Precision.HIGHEST))
ixs = jnp.argsort(e)
enull = e = e[ixs]
nnull = nnull = jnp.where(jnp.cumsum(e/sum(e))>tobj.stol)[0][0]
P_s = P_ = P[:,ixs[:nnull]]
#gamma_r = -jnp.linalg.pinv(yl.T.dot(yl)).dot(yl.sum(axis=0))
#gamma_r = gamma_r = jnp.linalg.pinv((yl.T.dot(yl))*jnp.reciprocal(len(yl))).dot(yl.mean(axis=0))
xo_ = xo_ = jnp.array([xo[i] - xo[j] for i in range(len(xo)) for j in range(i+1,len(xo))])
yavg  = yavg = jnp.array([yl[i]-yl[j] for i in range(len(yl)) for j in range(i+1,len(yl))])
# m = ((xo_**2).mean(axis=1)>1e-2)*((yavg**2).mean(axis=1)>1e-2)
# xo_ = xo = xo_[m,:]
# yavg = yavg = yavg[m,:]
Vg = Vg = xo_.dot(tobj.Uno)
Vl = Vl = jnp.array([(tobj.Unl.T).dot(jnp.diag(yavg[i])) for i in range(len(yavg))])
VlVl = VlVl = jnp.array([vl.T.dot(vl) for vl in Vl])
A = P_.T.dot(jnp.mean(VlVl,axis=0)).dot(P_)
Vlg = Vlg = -jnp.array([vl.T.dot(vg) for vl,vg in zip(Vl,Vg)])#jnp.array([vl.T.dot(vg+vl.dot(gamma_r)) for vl,vg in zip(Vl,Vg)])
#Vlg = Vlg = jnp.array([vl.T.dot(vg+vl.dot(gamma_r)) for vl,vg in zip(Vl,Vg)])
b = P_.T.dot(jnp.mean(Vlg,axis=0))

yl = jnp.vstack((datas[_][1][:,tobj.nobs:] for _ in [1]))#range(len(datas))))
xo = jnp.vstack((datas[_][1][:,:tobj.nobs] for _ in [1]))#range(len(datas))))
e1, P1 = jnp.linalg.eigh(jnp.cov(yl.T))

# +
Unl = tobj.Unl
Uno = tobj.Uno
# Unln = tobj.Unln_
# Unon = tobj.Unon_
u,s,vt=jnp.linalg.svd(Unl.dot(Unl.T))
Unln = u[:,s<1e-10]
u,s,vt=jnp.linalg.svd(Uno.dot(Uno.T))
Unon = u[:,s<1e-10]
Plo = -jnp.linalg.pinv(Unl.dot(Unl.T)).dot(Unl).dot(Uno.T)
Pol = -jnp.linalg.pinv(Uno.dot(Uno.T)).dot(Uno).dot(Unl.T)
Point = (Pol.dot(Plo))
Plint = (Plo.dot(Pol))
bo_ints = []
bl_ints = []
dxls = []
for d in range(2):
    #d = 0
    n = 20
    nobs = tobj.nobs
    t = datas0[d][0]
    xo = datas0[d][1][:,:nobs]
    xl = datas0[d][1][:,nobs:]
    dxo = jnp.diff(xo,axis=0)#xo-xo[n,:]
    dxl = jnp.diff(xl,axis=0)#xl-xl[n,:]
    dxol = dxo.dot(Point.T)
    dxlo = dxl.dot(Plint.T)
    dxoint = dxo.dot(jnp.eye(len(Point))-Point.T)
    bo_int = (dxoint).dot(Unon).dot(jnp.linalg.pinv((Unon.T.dot(Unon))))
    dxlint = dxl.dot(jnp.eye(len(Plint))-Plint.T)
    bl_int = (dxlint).dot(Unln).dot(jnp.linalg.pinv((Unln.T.dot(Unln))))
    plt.figure(dpi=100)
    plt.plot(bl_int);
    bo_ints += [bo_int]
    bl_ints += [bl_int]
    dxls += [dxlint]
    
plt.figure(dpi=100)
plt.plot(bl_int);
plt.figure(dpi=100)
plt.plot(dxlo+dxlint*0+bl_int.dot(Unln.T))
#plt.plot(dxo+xoint*0+bo_int.dot(Unon.T))
plt.figure(dpi=100)
plt.plot(dxl)
#plt.plot(dxo)
plt.figure()
plt.plot(datas0[0][0],datas0[0][1])
# -

# plt.imshow(tobj.sigms_p[0]);plt.colorbar();plt.figure();plt.figure()
# plt.imshow(tobj.sigms_p[1]);plt.colorbar();plt.figure()
# plt.imshow(tobj._cinv(tobj.sigms_p[0]));plt.colorbar();plt.figure()
# plt.imshow(tobj._cinv(tobj.sigms_p[1]));plt.colorbar()
# print(jnp.sqrt(jnp.diag(tobj._cinv(tobj.sigms_p[0]))))
# plt.figure()
for _ in [0,1]:
    plt.figure()
    plt.imshow(tobj.sigms[0][_]+jnp.diag(tobj.isigs[0][_]));plt.colorbar()
    plt.gca().set_title('sigma model')
    plt.figure()
    plt.imshow(tobj.sigms[1][_]+jnp.diag(tobj.isigs[1][_]));plt.colorbar()
    plt.gca().set_title('sigma data')
    plt.figure()
    # plt.imshow(tobj.omgas[1][1][_]);plt.colorbar()
    # plt.figure()
    plt.imshow(tobj.sigm_p[_]+jnp.diag(tobj.isigs[2][_]))
    plt.gca().set_title('sigma params')
    plt.colorbar()

plt.imshow(jnp.linalg.pinv(tobj.sigm_p[0])); plt.colorbar()
plt.figure()
plt.imshow(jnp.linalg.pinv(tobj.sigm_p[1])); plt.colorbar()

from trainer_source import *
from trainer_source import trainer as trainer_

tobj.dump('trainer_{}_{}{}'.format(tag,no,addon))

addon = '_alpha107sens_uq_full'
tobj.load('trainer_{}_{}{}.npz'.format(tag,no,addon))
pars_nn = tobj.params[0]
pars_m = tobj.params[1]
#pars_sc = tobj.params[2]

tobj.tol = 1e-20  
# adam_kwargs  = {'step_size':1e-4,'eps':1e-12,'b1':0.9,'b2':0.999}
# tobj.set_optimizer(optimizers.adam,adam_kwargs)
adam_kwargs  = {'step_size':1e-3,'eps':1e-22,'b1':0.9,'b2':0.999}
tobj.set_optimizer(optimizers.adam,adam_kwargs)
tobj.rejit()
tobj.num_iter = 300

# + active=""
# mode = 'inverse'
# for i in range(len(pars[no]['bc'])):
#             
#     dstack = jnp.vstack([_[1][:,pars[no]['nncpars'][i][mode]['predictor']['nobs']:] for _ in datas0])
#     s_ = jnp.std(dstack,axis=0)
#
#     t, x, [], dx = datas0[i]
#     #t_ = t[:-1,:]+jnp.diff(t,axis=0)/2
#     t_ = []
#
#     if tag == 'fwd': 
#         data = [(t,[],t_)]
#     elif tag == 'inv':
#         xinv  = x[:,:pars[no]['nncpars'][i][mode]['predictor']['nobs']] 
#         xinv  = xinv + random.normal(random.PRNGKey(0),xinv.shape)*0.00
#         xinv = xinv*(xinv>=0.)
#         data = [(t,xinv,t_)]
#     elif tag == 'invsc':
#         d = x[:,pars[no]['nncpars'][i][mode]['predictor']['nobs']:]
#         d = d/s_
#         #d = (d-d.mean(axis=0))/d.std(axis=0)
#         #d =  d+d.min(axis=0)
#         x_sc = jnp.hstack((x[:,:pars[no]['nncpars'][i][mode]['predictor']['nobs']],d))
#         data = [(t,x_sc,t_)]
#     elif tag == 'invvwn':
#         d = x[:,pars[no]['nncpars'][i][mode]['predictor']['nobs']:]
#         d = d/s_
#         #d = (d-d.mean(axis=0))/d.std(axis=0)
#         #d =  d+d.min(axis=0)
#         x_sc = jnp.hstack((x[:,:pars[no]['nncpars'][i][mode]['predictor']['nobs']],d))
#         #x_scwn = jnp.abs(x_sc + random.normal(random.PRNGKey(0),x_sc.shape)*0.025)
# #                 x_scwn = x_sc + jnp.hstack((random.normal(random.PRNGKey(0),x_sc[:,:pars[no]['nncpars'][i][mode]['predictor']['nobs']].shape)*0.01,
# #                                            random.normal(random.PRNGKey(0),x_sc[:,pars[no]['nncpars'][i][mode]['predictor']['nobs']:].shape)*0.01*jnp.sqrt(jnp.reciprocal(s_))))
#         x_scwn = x_sc + jnp.hstack((random.normal(random.PRNGKey(0),x_sc[:,:pars[no]['nncpars'][i][mode]['predictor']['nobs']].shape)*0.01,
#                                    random.normal(random.PRNGKey(0),x_sc[:,pars[no]['nncpars'][i][mode]['predictor']['nobs']:].shape)*0.005*jnp.reciprocal(jnp.mean(s_))))
#         #x_scwn = x_scwn*(x_scwn>=0.)
#         data = [(t,x_scwn,t_)]
#     datas  += data
# -

tobj.refresh = .99#.99
tobj.__setattr__('err_tags',['MLE MODEL','MLE DATA','MLE MODEL 2'])
while True:
    for i in range(100):
        tobj.num_epochs = 100
        
        tobj.train(datas,alpha=1.,isig=1.,beta=0.5)#1.)

_vvkl_div = vmap(tobj._kl_div,in_axes=(0,0))

A = random.uniform(random.PRNGKey(0),shape=[4,4])
S_ = random.uniform(random.PRNGKey(1),shape=[4,4])
S = S_.T.dot(S_)
B = A.T.dot(S).dot(A)
ea, Pa = jnp.linalg.eigh(A.T.dot(A))
B_ = Pa.dot(jnp.diag(jnp.sqrt(ea))).dot(S).dot((Pa.dot(jnp.diag(jnp.sqrt(ea)))).T)

jnp.sum(jnp.log(jnp.abs(jnp.linalg.eigvalsh(tobj.sigm_p[1]))))

jnp.sum(jnp.log(jnp.abs(jnp.linalg.eigvalsh(tobj.sigm_p[0]))))

plt.figure(dpi=100)
errs = jnp.array([_[3][1] for _ in tobj.iter_data[2:]])
plt.plot(errs[-0:,0],errs[-0:,1],'.-',alpha=0.25,ms=3,lw=0.2);# plt.gca().set_yscale('log'),plt.gca().set_xscale('log');
plt.gca().set_xlabel('$-\log{(\mathcal{L}_\dot{x})}$');plt.gca().set_ylabel('$-\log{(\mathcal{L}_{x})}$')
plt.gca().set_aspect(1.0/plt.gca().get_data_ratio(), adjustable='box')#.set_aspect('equal', adjustable='box')

jnp.exp(tobj.params[1][0])[::2]/jnp.exp(tobj.params[1][0])[1::2],pars[no]['kinpars'][::2]/pars[no]['kinpars'][1::2]

list(zip(pars[no]['kinpars'],jnp.exp(tobj.params[1][0])*(jnp.exp(tobj.params[1][0])>1e-10)))

plt.plot(pars[no]['kinpars'][:],jnp.exp(tobj.params[1][0])[:],'.')
#plt.plot(pars[no]['kinpars'][:],jnp.exp(tobj.iter_data[-5500][1][-1][-1][-1]),'.')
plt.gca().set_yscale('log'); plt.gca().set_xscale('log');
ymax = max([plt.gca().get_xbound()[1],plt.gca().get_ybound()[1]])
ymin = min([plt.gca().get_xbound()[0],plt.gca().get_ybound()[0]])
plt.plot([ymin,ymax],[ymin,ymax],'-')
plt.gca().set_xbound([ymin,ymax])
plt.gca().set_ybound([ymin,ymax])
jnp.linalg.norm(jnp.log(pars[no]['kinpars'])-tobj.params[1][0])
#jnp.linalg.norm((pars[no]['kinpars'])-jnp.exp(tobj.params[1][0]))

fig, axs = plt.subplots(4,int(len(datas0)),figsize=[15*len(datas0)/2.,20],dpi=75)
axs = axs[:,jnp.newaxis] if len(datas0)==1 else axs
errs = tobj.errs_fun(tobj.params,tobj._sparams(None,tobj.idxs0),datas)
for _ in range(len(datas0)):
    e0 = errs[1][_][:len(datas0[_][0]),3:]
    e1 = tobj.nn[_].batched_state(tobj.params[0][_],datas0[_][0])[:,3:]-datas0[_][1][:,3:]
    axs[0][_].plot(datas0[_][0],tobj.nn[_].batched_state(tobj.params[0][_],datas0[_][0])[:,:3],'--')
    #axs[0][_].set_xscale('log')
    #axs[0][_].set_yscale('log')
    ylim = axs[0][_].get_ylim()
    v = axs[0][_].twinx()
    #v.set_yscale('log')
    #v.plot(datas0[_][0],datas[_][1][:,:3])
    ls = v.plot(datas0[_][0],datas[_][1][:,:3],'-',alpha=1)
    v.set_ylim(ylim)
    [l.set_label(_) for l,_ in zip(ls,pars[no]['sps'][:3])]
    v.legend(fontsize=18)
    ls = axs[1][_].plot(datas0[_][0],tobj.nn[_].batched_state(tobj.params[0][_],datas0[_][0])[:,3:],'.-',ms=1.5)
    ylim = axs[1][_].get_ylim()
    v = axs[1][_].twinx()
    #v.plot(datas[_][0],datas[_][1][:,3:]*jnp.exp(tobj.params[2]))ii
    if tobj.scale:
        v.plot(datas[_][0],datas[_][1][:,3:]*(tobj.scales),'-o',ms=3.,lw=0.2)
        #v.plot(datas[_][0],datas[_][1][:,3:]*(s_),'',ms=1.)
    else:
        v.plot(datas[_][0],datas0[_][1][:,3:],'-',ms=1.)
    #v.plot(datas0[_][0],datas0[_][1][:,3:])
    [l.set_label(_) for l,_ in zip(ls,pars[no]['sps'][3:])]
    axs[1][_].legend(fontsize=18)
    v.set_ylim(ylim)
#     ls2_inf  = axs[2][_].plot(errs[1][_][:,3:].flatten(),(tobj.nn[1][_].batched_state(tobj.params[1][0][_],t)[:,:]\
#                               -datas0[_][1][:,3:]).flatten(),'.')
#     ls2_inf  = axs[2][_].plot((e0*e1).sum(axis=1)/(jnp.linalg.norm(e0,axis=1)*jnp.linalg.norm(e1,axis=1)),'.')
    ls2_inf  = axs[2][_].hist(((e0*e1).sum(axis=1)/(jnp.linalg.norm(e0,axis=1)*jnp.linalg.norm(e1,axis=1))).tolist())#,'.')
    ls3_inf  = axs[3][_].hist(e0.flatten().tolist(),bins=15);axs[3][_].hist(e1.flatten(),bins=15,alpha=0.5);
    v2       = axs[2][_].twinx()
    ylim2    = axs[2][_].get_ylim()
#     ls2      = v2.plot(t,tobj.nn[0][_].batched_state(tobj.params[0][0][_],t)[:,3:],'--')
    v2.set_ylim(ylim2)
plt.tight_layout()
plt.savefig('out.pdf',dpi=300)

pars[no]['sps'][3:]

['$k_{{{}}}$'.format(j+str(int(nads+i)))  for i in range(int(len(tobj.params[1][0][nads:])/2.)) for j in ['','-'] ]

linesp2

# +
from matplotlib.ticker import LogFormatterExponent

mpl.rcParams.update({'font.family':'serif'})
mpl.rcParams.update({'font.serif':'serif'})
#mpl.rcParams.update({'font.sans-serif':'Roboto'})
mpl.rcParams.update({'font.sans-serif':'FreeSerif'})
mpl.rcParams.update({'mathtext.fontset':'cm'})
mpl.rcParams.update({'mathtext.fallback_to_cm':True})
mpl.rcParams.update({'font.size':12})
mpl.rcParams.update({'axes.unicode_minus':False})
mpl.rcParams.update({'text.usetex':False})
mpl.rcParams.update({'legend.fontsize': 13.,
          'legend.handlelength': 1})
#fs=14.
fac=1.5
fac_ms=2.
fs=14.*fac
mpl.rcParams.update({   'axes.titlesize' : fs*11/12.,
                        'axes.labelsize' : fs*10/12.,
                        'lines.linewidth' : 1,
                        'lines.markersize' : fs*10/12.,
                        'xtick.labelsize' : fs*10/12.,
                        'ytick.labelsize' : fs*10/12.})
mpl.rcParams.update({'legend.handletextpad':.4,
                     'legend.handlelength':.6,
                      'legend.columnspacing':.5,
                      'legend.borderaxespad':.5})
mpl.rcParams['axes.linewidth'] = 0.75 #set the value globally

SMALL_SIZE = 12*fac
MEDIUM_SIZE = 12*fac
BIGGER_SIZE = 13*fac

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend',  fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

left  = 0.05  # the left side of the subplots of the figure
right = 0.925    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.95      # the top of the subplots of the figure
wspace = 0.225*1.2#*1.25*2.   # the amount of width reserved for blank space between subplots
hspace = 0.25*2#*0.75   # the amount of height reserved for white space between subplots

n = 1
nads = 6
t, x0 = datas[n][:2]
t = t[::2]
x0 = x0[::2,:]
#t, x0 = t[::3], x0[::3,:]
opt_inv = tobj.iter_data#[:10]
#fig, axs = plt.subplots(2,3,figsize=[13.5*1.15,10*0.45*2],dpi=100)
#fig, axs = plt.subplots(1,4,figsize=[11*2./3*1.5+1.25,10*0.35*2*0.9],dpi=150)
fig, axs = plt.subplots(1,4,figsize=[4.75,22.5][::-1],dpi=150)
#[_.set_aspect('equal', adjustable='box') for _ in axs]
#axs = axs.tolist()
#plt.sca(axs[0])
lines11 = axs[0].plot(t, x0[:,:3],'-o',alpha=0.7,ms=fac_ms*2)
lines21 = [axs[0].plot([],[],'-o',lw=0.2,\
                  ms=fac_ms*4,markerfacecolor='None',markeredgecolor='black',alpha=0.7)[0] for _ in range(3)]
for _1,_2 in zip(lines11,lines21):
    _2.set_markeredgecolor(_1.get_markeredgecolor())
    _2.set_color(_1.get_color())
axs[0].set_xlabel('')

lines12 = axs[1].plot(t, x0[:,3:]*s_,'-o',alpha=0.7,ms=fac_ms*2)
lines22 = [axs[1].plot([],[],'-o',lw=0.2,\
                  ms=fac_ms*4,markerfacecolor='None',markeredgecolor='black',alpha=0.7)[0] for _ in range(len(tobj.model.M)-tobj.nobs)]
for _1,_2 in zip(lines12,lines22):
    _2.set_markeredgecolor(_1.get_markeredgecolor())
    _2.set_color(_1.get_color())

#leg12 = axs[0].legend(iter(lines12+lines22), [_+__ for _ in pars[no]['sps'][3:] for __ in ['$_{NUM}$','$_{NN}$']],loc=7)
#leg12 = axs[0].legend(iter(lines12), [_+__ for _ in pars[no]['sps'][:3] for __ in ['$_{NUM}$']],loc=2)
#axs[1][0].set_title('State Variables')
##axs[1][0].set_xlabel('Time')
##axs[1][0].set_ylabel('C');
linesp1 = [axs[2].plot([],[],'o',lw=0.2,ms=fac_ms*3,alpha=0.6)[0] for _ in range(tobj.nobs)]
leg21 = axs[2].legend(linesp1,pars[no]['sps'][:3],loc=4,labelspacing=0.3,columnspacing=0.3)
['A$_{NUM}$', 'B$_{NUM}$', 'C$_{NUM}$','A$_{NN}$','B$_{NN}$','C$_{NN}$']
linesp2 = [axs[2].plot([],[],'o',lw=0.2,mew=1.75,ms=fac_ms*4,alpha=0.65,mfc='None')[0] for _ in range(len(tobj.model.M)-tobj.nobs)]
[_.set_markeredgecolor('C'+str(i)) for i,_ in enumerate(linesp2)]
leg22 = axs[2].legend(linesp2,pars[no]['sps'][3:],loc=0,ncol=1, bbox_to_anchor=(0.23, 1.),labelspacing=0.3,columnspacing=0.3)
axs[2].add_artist(leg21)
axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))

axs[3].set_xlabel('')
lines31 = [axs[3].plot([], [],'.',alpha=0.85,ms=fac_ms*7)[0] for _ in range(len(tobj.params[1][0][:nads]))]
j = 0
for _ in range(len(lines31)):
    lines31[_].set_color('C'+str(j))
    if _%2==1:
        j+=1
lleg1 = axs[3].legend(iter(lines31), ['$k_{{{}}}$'.format(j+str(int(i)+1)) for i in range(int(len(tobj.params[1][0][:nads])/2.))  for j in ['','-']],loc='lower left',ncol=3, bbox_to_anchor=(0., .75),labelspacing=0.2,columnspacing=0.3)
#axs[0][2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
lines31_ = axs[3].plot([],[],'-',lw=0.5,alpha=0.5,c='grey')
lines32 = [axs[3].plot([], [],'.',alpha=0.95,ms=fac_ms*7,mew=1.75)[0] for _ in range(len(tobj.params[1][0][nads:]))]
axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))
j=6
for _ in range(len(lines32)):
    lines32[_].set_markeredgecolor('C'+str(j))
    lines32[_].set_color('None')
    if _%2==1:
        j+=1
lleg2 = axs[3].legend(iter(lines32), ['$k_{{{}}}$'.format(j+str(int(nads/2)+i+1)) for i in range(int(len(tobj.params[1][0][nads:])/2.))  for j in ['','-']],loc='lower left',ncol=4,\
                      bbox_to_anchor=(0.275, -.025),labelspacing=0.2,columnspacing=0.3)
axs[3].add_artist(lleg1)
axs[3].xaxis.set_major_formatter(LogFormatterExponent())
axs[3].yaxis.set_major_formatter(LogFormatterExponent())
axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[3].set_xscale('log')
axs[3].set_yscale('log')


title = fig.suptitle('')
plt.tight_layout(rect=[0.,0.,1.,0.95])
plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

epochs = [0,1,2]
while epochs[-1]<len(opt_inv)-1:
    epochs += [int((epochs[-1]+1)**1.025)]
epochs[-1] = len(opt_inv[-1])-1
#pack = list(zip(epochs,[opt_inv[-1][int(_)] for _ in epochs]))
pack = list(zip(epochs,[[opt_inv[int(_)][1][0][0],opt_inv[int(_)][1][0][-1]] for _ in epochs]))
#pack = [[opt_inv[int(_)][0], [opt_inv[int(_)][1][0][n],opt_inv[int(_)][1][0][-1]]] for _ in epochs]

# def init():
axs[0].set_ylim([-0.025,0.625]) 
#axs[0].set_ylim([-0.025,1.025]) 
axs[0].set_xlim([-0.025,1.025])
axs[2].set_ylim([-8,8])
axs[2].set_xlim([-8,8])
axs[1].set_ylim([-0.025,1.05])
axs[1].set_xlim([-0.025,1.025])
axs[2].plot([-8,8],[-8,8],'-',lw=0.5,alpha=0.5,c='grey')
#     return lines21+linesp1+lines31+lines22+linesp2+lines32#+axs

i=-2
#def animate(i,lobjs,pack):
    #lines21,linesp1,lines31,lines22,linesp2,lines32,lines31_,lines32_ = lobjs
epoch, params = pack[i]
nn_params, model_params = params
#title.set_text('epoch {:03d}'.format(int(epoch)))
d = tobj.nn[n].batched_state(nn_params[n], t)
for _ in range(tobj.nobs):
    lines21[_].set_data(t,d[:,_])
for _ in range(tobj.nobs,len(tobj.model.M)):
    lines22[_-tobj.nobs].set_data(t,d[:,_])
x_parity = tobj.model.batched_eval([jnp.log(pars[4]['kinpars'])],[t,jnp.hstack((x0[:,:tobj.nobs],x0[:,tobj.nobs:]*s_))])#[:,:,0]
y_parity = tobj.model.batched_eval(model_params,[t,tobj.nn[n].batched_state(nn_params[n],t)])#[:,:,0]
x_parity, y_parity = [(_-_.mean(axis=0))/_.std(axis=0) for _ in [x_parity, y_parity]]
for _ in range(tobj.nobs):
    linesp1[_].set_data(x_parity[:,_],y_parity[:,_])
for _ in range(tobj.nobs,len(tobj.model.M)):
    linesp2[_-tobj.nobs].set_data(x_parity[:,_],y_parity[:,_])
for _ in range(len(tobj.params[1][0][:nads])):
    lines31[_].set_data((pars[4]['kinpars'])[_],jnp.exp(opt_inv[epoch][1][0][-1][0][_]))
for _ in range(nads,nads+len(tobj.params[1][0][nads:])):
    lines32[_-nads].set_data((pars[4]['kinpars'])[_],jnp.exp(opt_inv[epoch][1][0][-1][0][_]))
min31 = jnp.min(jnp.array([jnp.min(jnp.array([_.get_data()[0],_.get_data()[1]])) for _ in lines31+lines32]))
max31 = jnp.max(jnp.array([jnp.max(jnp.array([_.get_data()[0],_.get_data()[1]])) for _ in lines31+lines32]))
bds1 = [1.5*min31 if min31<0 else 0.85*min31,1.5*max31 if max31>0 else 0.85*max31]
axs[3].set_xlim(bds1)
axs[3].set_ylim(bds1)
lines31_[0].set_data(bds1,bds1)
#axs[0].set_aspect(1.05/0.65, adjustable='box')
plt.savefig('fitting1.svg',dpi=300,bbox_inches='tight')


#plt.tight_layout()
#plt.tight_layout()
#dots.set_offsets( u[:,[0,i]])
#line.set_data(u[:,0],u[:,i])
#return lines21+linesp1+lines31+lines22+linesp2+lines32+lines31_+lines32_#+axs

#init()
#animate(-1,[lines21,linesp1,lines31,lines22,linesp2,lines32,lines31_,lines32_],pack);

# +
from matplotlib.ticker import LogFormatterExponent

mpl.rcParams.update({'font.family':'serif'})
mpl.rcParams.update({'font.serif':'serif'})
#mpl.rcParams.update({'font.sans-serif':'Roboto'})
mpl.rcParams.update({'font.sans-serif':'FreeSerif'})
mpl.rcParams.update({'mathtext.fontset':'cm'})
mpl.rcParams.update({'mathtext.fallback_to_cm':True})
mpl.rcParams.update({'font.size':12})
mpl.rcParams.update({'axes.unicode_minus':False})
mpl.rcParams.update({'text.usetex':False})
mpl.rcParams.update({'legend.fontsize': 13.,
          'legend.handlelength': 1})
#fs=14.
fac=1.5
fac_ms=2.
fs=14.*fac
mpl.rcParams.update({   'axes.titlesize' : fs*11/12.,
                        'axes.labelsize' : fs*10/12.,
                        'lines.linewidth' : 1,
                        'lines.markersize' : fs*10/12.,
                        'xtick.labelsize' : fs*10/12.,
                        'ytick.labelsize' : fs*10/12.})
mpl.rcParams.update({'legend.handletextpad':.4,
                     'legend.handlelength':.6,
                      'legend.columnspacing':.5,
                      'legend.borderaxespad':.5})
mpl.rcParams['axes.linewidth'] = 0.75 #set the value globally

SMALL_SIZE = 12*fac
MEDIUM_SIZE = 12*fac
BIGGER_SIZE = 13*fac

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend',  fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

left  = 0.05  # the left side of the subplots of the figure
right = 0.925    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.95      # the top of the subplots of the figure
wspace = 0.225*1.2#*1.25*2.   # the amount of width reserved for blank space between subplots
hspace = 0.25*2#*0.75   # the amount of height reserved for white space between subplots

n = 1
nads = 6
t, x0 = datas[n][:2]
t = t[::2]
x0 = x0[::2,:]
#t, x0 = t[::3], x0[::3,:]
opt_inv = tobj.iter_data#[:10]
#fig, axs = plt.subplots(2,3,figsize=[13.5*1.15,10*0.45*2],dpi=100)
#fig, axs = plt.subplots(1,4,figsize=[11*2./3*1.5+1.25,10*0.35*2*0.9],dpi=150)
fig, axs = plt.subplots(1,4,figsize=[4.75,22.5][::-1],dpi=150)
#[_.set_aspect('equal', adjustable='box') for _ in axs]
#axs = axs.tolist()
#plt.sca(axs[0])
lines11 = axs[0].plot(t, (x0[:,:]*jnp.concatenate((jnp.array([1.]*3),s_))).dot(tobj.model.Ur),'-o',alpha=0.7,ms=fac_ms*2)
lines21 = [axs[0].plot([],[],'-o',lw=0.2,\
                  ms=fac_ms*4,markerfacecolor='None',markeredgecolor='black',alpha=0.7)[0] for _ in range(tobj.model.Ur.shape[1])]
for _1,_2 in zip(lines11,lines21):
    _2.set_markeredgecolor(_1.get_markeredgecolor())
    _2.set_color(_1.get_color())
axs[0].set_xlabel('')

lines12 = axs[1].plot(t, (x0[:,:]*jnp.concatenate((jnp.array([1.]*3),s_))).dot(tobj.model.Un),'-o',alpha=0.7,ms=fac_ms*2)
lines22 = [axs[1].plot([],[],'-o',lw=0.2,\
                  ms=fac_ms*4,markerfacecolor='None',markeredgecolor='black',alpha=0.7)[0] for _ in range(tobj.model.Un.shape[1])]
for _1,_2 in zip(lines12,lines22):
    _2.set_markeredgecolor(_1.get_markeredgecolor())
    _2.set_color(_1.get_color())

#leg12 = axs[0].legend(iter(lines12+lines22), [_+__ for _ in pars[no]['sps'][3:] for __ in ['$_{NUM}$','$_{NN}$']],loc=7)
#leg12 = axs[0].legend(iter(lines12), [_+__ for _ in pars[no]['sps'][:3] for __ in ['$_{NUM}$']],loc=2)
#axs[1][0].set_title('State Variables')
##axs[1][0].set_xlabel('Time')
##axs[1][0].set_ylabel('C');
linesp1 = [axs[2].plot([],[],'o',lw=0.2,ms=fac_ms*3,alpha=0.6)[0] for _ in range(tobj.model.Ur.shape[1])]
leg21 = axs[2].legend(linesp1,['$z^R_{{{}}}$'.format(_+1) for _ in range(len(linesp1))],loc=0,labelspacing=0.3,columnspacing=0.3,ncol=4)
linesp2 = [axs[3].plot([],[],'o',lw=0.2,mew=1.75,ms=fac_ms*4,alpha=0.65,mfc='None')[0] for _ in range(tobj.model.Un.shape[1])]
[_.set_markeredgecolor('C'+str(i)) for i,_ in enumerate(linesp2)]
leg22 = axs[3].legend(linesp2,['$z^N_{{{}}}$'.format(_+1) for _ in range(len(linesp2))],loc=0,ncol=1, bbox_to_anchor=(0.23, 1.),labelspacing=0.3,columnspacing=0.3)
axs[2].add_artist(leg21)
axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))

# axs[3].set_xlabel('')
# lines31 = [axs[3].plot([], [],'.',alpha=0.85,ms=fac_ms*7)[0] for _ in range(len(tobj.params[1][0][:nads]))]
# j = 0
# for _ in range(len(lines31)):
#     lines31[_].set_color('C'+str(j))
#     if _%2==1:
#         j+=1
# lleg1 = axs[3].legend(iter(lines31), ['$k_{{{}}}$'.format(j+str(int(i)+1)) for i in range(int(len(tobj.params[1][0][:nads])/2.))  for j in ['','-']],loc='lower left',ncol=3, bbox_to_anchor=(0., .75),labelspacing=0.2,columnspacing=0.3)
# #axs[0][2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# lines31_ = axs[3].plot([],[],'-',lw=0.5,alpha=0.5,c='grey')
# lines32 = [axs[3].plot([], [],'.',alpha=0.95,ms=fac_ms*7,mew=1.75)[0] for _ in range(len(tobj.params[1][0][nads:]))]
# axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
# axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))
# j=6
# for _ in range(len(lines32)):
#     lines32[_].set_markeredgecolor('C'+str(j))
#     lines32[_].set_color('None')
#     if _%2==1:
#         j+=1
# lleg2 = axs[3].legend(iter(lines32), ['$k_{{{}}}$'.format(j+str(int(nads/2)+i+1)) for i in range(int(len(tobj.params[1][0][nads:])/2.))  for j in ['','-']],loc='lower left',ncol=4,\
#                       bbox_to_anchor=(0.275, -.025),labelspacing=0.2,columnspacing=0.3)
# axs[3].add_artist(lleg1)
# # axs[3].xaxis.set_major_formatter(LogFormatterExponent())
# # axs[3].yaxis.set_major_formatter(LogFormatterExponent())
# axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
# axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))
# axs[3].set_xscale('log')
# axs[3].set_yscale('log')


title = fig.suptitle('')
plt.tight_layout(rect=[0.,0.,1.,0.95])
plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

epochs = [0,1,2]
while epochs[-1]<len(opt_inv)-1:
    epochs += [int((epochs[-1]+1)**1.025)]
epochs[-1] = len(opt_inv[-1])-1
#pack = list(zip(epochs,[opt_inv[-1][int(_)] for _ in epochs]))
pack = list(zip(epochs,[[opt_inv[int(_)][1][0][0],opt_inv[int(_)][1][0][-1]] for _ in epochs]))
#pack = [[opt_inv[int(_)][0], [opt_inv[int(_)][1][0][n],opt_inv[int(_)][1][0][-1]]] for _ in epochs]

# def init():
#axs[0].set_ylim([-0.025,1.025])#0.625]) 7
#axs[0].set_ylim([-0.025,1.025]) 
axs[0].set_xlim([-0.025,1.025])
axs[2].set_ylim([-4,1])
axs[2].set_xlim([-4,1])
axs[3].set_ylim([-1e-15,1e-15])
axs[3].set_xlim([-1e-15,1e-15])
#axs[1].set_ylim([-0.025,1.05])
axs[1].set_xlim([-0.025,1.025])
axs[2].plot([-4,1],[-4,1],'-',lw=0.5,alpha=0.5,c='grey')
axs[3].plot([-3,3],[-3,3],'-',lw=0.5,alpha=0.5,c='grey')
#     return lines21+linesp1+lines31+lines22+linesp2+lines32#+axs

i=-2
#def animate(i,lobjs,pack):
    #lines21,linesp1,lines31,lines22,linesp2,lines32,lines31_,lines32_ = lobjs
epoch, params = pack[i]
nn_params, model_params = params
#title.set_text('epoch {:03d}'.format(int(epoch)))
dr = tobj.nn[n].batched_state(nn_params[n], t).dot(tobj.model.Ur)
dn = tobj.nn[n].batched_state(nn_params[n], t).dot(tobj.model.Un)
for _ in range(tobj.model.Ur.shape[1]):
    lines21[_].set_data(t,dr[:,_])
for _ in range(tobj.model.Un.shape[1]):
    lines22[_].set_data(t,dn[:,_])
mdx = tobj.model.batched_eval([jnp.log(pars[4]['kinpars'])],[t,jnp.hstack((x0[:,:tobj.nobs],x0[:,tobj.nobs:]*s_))])
mdx_ = tobj.model.batched_eval(model_params,[t,tobj.nn[n].batched_state(nn_params[n],t)])
x_parity_r = mdx.dot(tobj.model.Ur)#[:,:,0]
y_parity_r = mdx_.dot(tobj.model.Ur)#[:,:,0]
x_parity_n = mdx.dot(tobj.model.Un)#[:,:,0]
y_parity_n = mdx_.dot(tobj.model.Un)#[:,:,0]
x_parity, y_parity = [(_-_.mean(axis=0))/_.std(axis=0) for _ in [x_parity, y_parity]]
for _ in range(tobj.model.Ur.shape[1]):
    linesp1[_].set_data(x_parity_r[:,_],y_parity_r[:,_])
for _ in range(tobj.model.Un.shape[1]):
    linesp2[_].set_data(x_parity_n[:,_],y_parity_n[:,_])
# for _ in range(len(tobj.params[1][0][:nads])):
#     lines31[_].set_data((pars[4]['kinpars'])[_],jnp.exp(opt_inv[epoch][1][0][-1][0][_]))
# for _ in range(nads,nads+len(tobj.params[1][0][nads:])):
#     lines32[_-nads].set_data((pars[4]['kinpars'])[_],jnp.exp(opt_inv[epoch][1][0][-1][0][_]))
# min31 = jnp.min(jnp.array([jnp.min(jnp.array([_.get_data()[0],_.get_data()[1]])) for _ in lines31+lines32]))
# max31 = jnp.max(jnp.array([jnp.max(jnp.array([_.get_data()[0],_.get_data()[1]])) for _ in lines31+lines32]))
# bds1 = [1.5*min31 if min31<0 else 0.85*min31,1.5*max31 if max31>0 else 0.85*max31]
# axs[3].set_xlim(bds1)
# axs[3].set_ylim(bds1)
# lines31_[0].set_data(bds1,bds1)
#axs[0].set_aspect(1.05/0.65, adjustable='box')
plt.savefig('fittingrn.svg',dpi=300,bbox_inches='tight')


#plt.tight_layout()
#plt.tight_layout()
#dots.set_offsets( u[:,[0,i]])
#line.set_data(u[:,0],u[:,i])
#return lines21+linesp1+lines31+lines22+linesp2+lines32+lines31_+lines32_#+axs

#init()
#animate(-1,[lines21,linesp1,lines31,lines22,linesp2,lines32,lines31_,lines32_],pack);
# -

381,392

# +
mpl.rcParams.update({'font.family':'serif'})
mpl.rcParams.update({'font.serif':'Consolas'})
mpl.rcParams.update({'font.sans-serif':'Consolas'})
#mpl.rcParams.update({'font.sans-serif':'FreeSerif'})
# mpl.rcParams.update({'mathtext.fontset':'cm'})
# mpl.rcParams.update({'mathtext.fallback_to_cm':True})
mpl.rcParams.update({'font.size':12})
mpl.rcParams.update({'axes.unicode_minus':False})
mpl.rcParams.update({'text.usetex':False})
mpl.rcParams.update({'legend.fontsize': 13.,
          'legend.handlelength': 1})
fs=14.
mpl.rcParams.update({   'axes.titlesize' : fs*11/12.,
                        'axes.labelsize' : fs*10/12.,
                        'lines.linewidth' : 1,
                        'lines.markersize' : fs*10/12.,
                        'xtick.labelsize' : fs*10/12.,
                        'ytick.labelsize' : fs*10/12.})
mpl.rcParams.update({'legend.handletextpad':.4,
                     'legend.handlelength':.6,
                      'legend.columnspacing':.5,
                      'legend.borderaxespad':.5})
mpl.rcParams['axes.linewidth'] = 0.75 #set the value globally

SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 13

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend',  fontsize=SMALL_SIZE*10./12.)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

left  = 0.1  # the left side of the subplots of the figure
right = 0.925    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.875      # the top of the subplots of the figure
wspace = 0.225*1.35   # the amount of width reserved for blank space between subplots
hspace = 0.25*0.75   # the amount of height reserved for white space between subplots

n = 1
nads = 6
t, x0 = datas[n][:2]
#t, x0 = t[::3], x0[::3,:]
opt_inv = tobj.iter_data#[:10]
#fig, axs = plt.subplots(2,3,figsize=[13.5*1.15,10*0.45*2],dpi=100)
fig, axs = plt.subplots(2,3,figsize=[10,6],dpi=125)
#axs = axs.tolist()
#plt.sca(axs[0])
lines11 = axs[0][0].plot(t, x0[:,:3],'-o',alpha=0.5,ms=2)
lines21 = [axs[0][0].plot([],[],'-o',lw=0.2,\
                  ms=4,markerfacecolor='None',markeredgecolor='black',alpha=0.5)[0] for _ in range(3)]
for _1,_2 in zip(lines11,lines21):
    _2.set_markeredgecolor(_1.get_markeredgecolor())
    _2.set_color(_1.get_color())
#['A$_{NUM}$', 'B$_{NUM}$', 'C$_{NUM}$','A$_{NN}$','B$_{NN}$','C$_{NN}$']
#leg11 = axs[0][0].legend(iter(lines11+lines21), [_+__ for _ in pars[no]['sps'][:3] for __ in ['$_{NUM}$','$_{NN}$']],loc=7)
##axs[0][0].set_title('State Variables')
axs[0][0].set_xlabel('')
##axs[0][0].set_ylabel('C');
linesp1 = [axs[0][1].plot([],[],'o',lw=0.2,ms=3,alpha=0.4)[0] for _ in range(tobj.nobs)]
leg21 = axs[0][1].legend(pars[no]['sps'][:3],loc=4,labelspacing=0.3,columnspacing=0.3)
##axs[0][1].set_title('Whittened Derivatives Parity Plot')
axs[0][1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0][1].yaxis.set_major_locator(MaxNLocator(integer=True))
##axs[0][1].set_ylabel('ANN');
##axs[0][2].set_title('Coupled Regression')
axs[0][2].set_xlabel('')
##axs[0][2].set_ylabel('Model Parameters')
lines31 = [axs[0][2].plot([], [],'.-',alpha=0.85,ms=7)[0] for _ in range(len(tobj.params[1][0][:nads]))]
#c = ['#d62728','#8c564b']
j = 0
for _ in range(len(lines31)):
    lines31[_].set_color('C'+str(j))
    if _%2==1:
        j+=1
#axs[0][2].plot([0,200],[1,1],'-',lw=0.5,alpha=0.2,color='blue')
#axs[0][2].legend(iter(lines31), ['$k_1$','$k_2$'],loc=4)
axs[0][2].legend(iter(lines31), ['$k_{{{}}}$'.format(j+str(int(i)+1)) for i in range(int(len(tobj.params[1][0][:nads])/2.))  for j in ['','-']],loc='lower left',ncol=3, bbox_to_anchor=(0., .725),labelspacing=0.2,columnspacing=0.3)
axs[0][2].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0][2].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[0][2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
lines31_ = axs[0][2].plot([],[],'-',lw=0.5,alpha=0.5,c='grey')

lines12 = axs[1][0].plot(t, x0[:,3:]*s_,'-o',alpha=0.5,ms=2)
lines22 = [axs[1][0].plot([],[],'-o',lw=0.2,\
                  ms=4,markerfacecolor='None',markeredgecolor='black',alpha=0.5)[0] for _ in range(len(tobj.model.M)-tobj.nobs)]
for _1,_2 in zip(lines12,lines22):
    _2.set_markeredgecolor(_1.get_markeredgecolor())
    _2.set_color(_1.get_color())
['A$_{NUM}$', 'B$_{NUM}$', 'C$_{NUM}$','A$_{NN}$','B$_{NN}$','C$_{NN}$']
#leg12 = axs[1][0].legend(iter(lines12+lines22), [_+__ for _ in pars[no]['sps'][3:] for __ in ['$_{NUM}$','$_{NN}$']],loc=7)
#axs[1][0].set_title('State Variables')
##axs[1][0].set_xlabel('Time')
##axs[1][0].set_ylabel('C');
linesp2 = [axs[1][1].plot([],[],'o',lw=0.2,ms=4,alpha=0.3)[0] for _ in range(len(tobj.model.M)-tobj.nobs)]
leg22 = axs[1][1].legend(pars[no]['sps'][3:],loc=0,ncol=1, bbox_to_anchor=(0.2225, 1.),labelspacing=0.3,columnspacing=0.3)
axs[1][1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1][1].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[1][1].set_title('Whittened Derivatives Parity Plot')
axs[1][1].set_xlabel('$f(\mathbf{x}(t))$')
axs[1][1].set_ylabel('$\dot{\mathbf{x}}_{s}(t)$');
axs[0][1].set_ylabel('$\dot{\mathbf{x}}_{g}(t)$');
axs[1][0].set_xlabel('Normalized Time - $t$');
axs[1][0].set_ylabel('${\mathbf{x}_{s}}(t)$');
axs[0][0].set_ylabel('${\mathbf{x}_{g}}(t)$');
axs[1][2].set_xlabel('Ground Truth - $ln(\mathbf{k})$')
ax02 = axs[0][2].set_ylabel('$ln(\mathbf{k}_g)$')
ax12 = axs[1][2].set_ylabel('$ln(\mathbf{k}_s)$')
pos02 = ax02.get_position()
pos12 = ax12.get_position()
axs[0][0].set_title('State Interpolation');
axs[0][1].set_title('Derivative-matching - PINN');
axs[0][2].set_title('Inverse Problem - Regression');

axs[0][0].get_shared_x_axes().join(axs[1][0],axs[0][0])
lines32 = [axs[1][2].plot([], [],'.-',alpha=0.85,ms=7)[0] for _ in range(len(tobj.params[1][0][nads:]))]
#c = ['#d62728','#8c564b']
j=6
for _ in range(len(lines32)):
    lines32[_].set_color('C'+str(j))
    if _%2==1:
        j+=1
#axs[1][2].plot([0,200],[1,1],'-',lw=0.5,alpha=0.2,color='blue')
axs[1][2].legend(iter(lines32), ['$k_{{{}}}$'.format(j+str(int(nads/2)+i+1)) for i in range(int(len(tobj.params[1][0][nads:])/2.))  for j in ['','-']],loc='lower left',ncol=4, \
                 bbox_to_anchor=(0., .725),labelspacing=0.2,columnspacing=0.3)
axs[1][2].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1][2].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[1][2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
lines32_ = axs[1][2].plot([],[],'-',lw=0.5,alpha=0.5,c='grey')


title = fig.suptitle('')
plt.tight_layout(rect=[0.,0.,1.,0.95])
plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

epochs = [0,1,2]
while epochs[-1]<len(opt_inv)-1:
    epochs += [int((epochs[-1]+1)**1.025)]
epochs[-1] = len(opt_inv[-1])-1
#pack = list(zip(epochs,[opt_inv[-1][int(_)] for _ in epochs]))
pack = list(zip(epochs,[[opt_inv[int(_)][1][0][0],opt_inv[int(_)][1][0][-1]] for _ in epochs]))
#pack = [[opt_inv[int(_)][0], [opt_inv[int(_)][1][0][n],opt_inv[int(_)][1][0][-1]]] for _ in epochs]

def init():
    axs[0][0].set_ylim([-0.025,0.625]) 
    axs[0][0].set_xlim([-0.025,1.025])
    axs[0][1].set_ylim([-8,8])
    axs[0][1].set_xlim([-8,8])
    axs[1][0].set_ylim([-0.025,1.05])
    axs[1][0].set_xlim([-0.025,1.025])
    axs[1][1].set_ylim([-8,8])
    axs[1][1].set_xlim([-8,8])
    axs[0][1].plot([-8,8],[-8,8],'-',lw=0.5,alpha=0.2,c='grey')
    axs[1][1].plot([-8,8],[-8,8],'-',lw=0.5,alpha=0.2,c='grey')
    return lines21+linesp1+lines31+lines22+linesp2+lines32+[axs[0][2].yaxis.label,axs[1][2].yaxis.label]

def animate(i):
    epoch, params = pack[i]
    nn_params, model_params = params
    title.set_text('epoch {:03d}'.format(int(epoch)))
    d = tobj.nn[n].batched_state(nn_params[n], t)
    for _ in range(tobj.nobs):
        lines21[_].set_data(t,d[:,_])
    for _ in range(tobj.nobs,len(tobj.model.M)):
        lines22[_-tobj.nobs].set_data(t,d[:,_])
    x_parity = tobj.model.batched_eval([jnp.log(pars[4]['kinpars'])],[t,jnp.hstack((x0[:,:tobj.nobs],x0[:,tobj.nobs:]*s_))])#[:,:,0]
    y_parity = tobj.model.batched_eval(model_params,[t,tobj.nn[n].batched_state(nn_params[n],t)])#[:,:,0]
    x_parity, y_parity = [(_-_.mean(axis=0))/_.std(axis=0) for _ in [x_parity, y_parity]]
    for _ in range(tobj.nobs):
        linesp1[_].set_data(x_parity[:,_],y_parity[:,_])
    for _ in range(tobj.nobs,len(tobj.model.M)):
        linesp2[_-tobj.nobs].set_data(x_parity[:,_],y_parity[:,_])
    for _ in range(len(tobj.params[1][0][:nads])):
        lines31[_].set_data(jnp.log(pars[4]['kinpars'])[_],opt_inv[epoch][1][0][-1][0][_])
    for _ in range(nads,nads+len(tobj.params[1][0][nads:])):
        lines32[_-nads].set_data(jnp.log(pars[4]['kinpars'])[_],opt_inv[epoch][1][0][-1][0][_])
    #axs[0][2].set_xlim([-1,epoch])
#     axs[0][2].set_xlim([1.1*jnp.min(jnp.array([jnp.min(jnp.array(_.get_data()[0])) for _ in lines31])),
#                      1.1*jnp.max(jnp.array([jnp.max(jnp.array(_.get_data()[0])) for _ in lines31]))])
#     axs[0][2].set_ylim([1.1*jnp.min(jnp.array([jnp.min(jnp.array(_.get_data()[1])) for _ in lines31])),
#                      1.1*jnp.max(jnp.array([jnp.max(jnp.array(_.get_data()[1])) for _ in lines31]))])
#     #axs[1][2].set_xlim([-1,epoch])
#     axs[1][2].set_xlim([1.1*jnp.min(jnp.array([jnp.min(jnp.array(_.get_data()[0])) for _ in lines32])),
#                      1.1*jnp.max(jnp.array([jnp.max(jnp.array(_.get_data()[0])) for _ in lines32]))])
#     axs[1][2].set_ylim([0.9*jnp.min(jnp.array([jnp.min(jnp.array(_.get_data()[1])) for _ in lines32])),
#                      1.1*jnp.max(jnp.array([jnp.max(jnp.array(_.get_data()[1])) for _ in lines32]))])
    min31 = jnp.min(jnp.array([jnp.min(jnp.array([_.get_data()[0],_.get_data()[1]])) for _ in lines31]))
    max31 = jnp.max(jnp.array([jnp.max(jnp.array([_.get_data()[0],_.get_data()[1]])) for _ in lines31]))
    bds1 = [1.1*min31 if min31<0 else 0.9*min31,1.1*max31 if max31>0 else 0.9*max31]
    axs[0][2].set_xlim(bds1)
    axs[0][2].set_ylim(bds1)
    lines31_[0].set_data(bds1,bds1)
    min32 = jnp.min(jnp.array([jnp.min(jnp.array([_.get_data()[0],_.get_data()[1]])) for _ in lines32]))
    max32 = jnp.max(jnp.array([jnp.max(jnp.array([_.get_data()[0],_.get_data()[1]])) for _ in lines32]))
    bds2 = [1.1*min32 if min32<0 else 0.9*min32,1.1*max32 if max32>0 else 0.9*max32]
    axs[1][2].set_xlim(bds2)
    axs[1][2].set_ylim(bds2)
    lines32_[0].set_data(bds2,bds2)
    axs[0][2].yaxis.label.set_position(pos02)
    axs[1][2].yaxis.label.set_position(pos12)
    #dots.set_offsets( u[:,[0,i]])
    #line.set_data(u[:,0],u[:,i])
    return lines21+linesp1+lines31+lines22+linesp2+lines32+lines31_+lines32_+[axs[0][2].yaxis.label,axs[1][2].yaxis.label]

# Call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(epochs), interval=125, blit=True)

plt.close(anim._fig)

# Call function to display the animation
HTML(anim.to_html5_video())

anim.save('kin4.gif', writer='imagemagick')

from IPython.display import Image
with open('kin4.gif','rb') as file:
    display(Image(file.read()))
    
# print('True parameters:      {}'.format(['{:.5f}'.format(float(_)) for _ in model_params0[0]]))
# print('Regressed parameters: {}'.format(['{:.5f}'.format(float(_)) for _ in model_params_inv[0]]))
# -


ax.xaxis.label.set_position

fig, ax = plt.subplots()
ax.plot([1,2],[2,3])
xlabel = plt.gca().set_xlabel('$\dot{\mathbf{x}}_{g}$')
ax.xaxis.label.set_position([0.,0.3])

# +
n = 1
nads = 6
t, x0 = datas[n][:2]
#t, x0 = t[::3], x0[::3,:]
opt_inv = tobj.iter_data#[:20]
#fig, axs = plt.subplots(2,3,figsize=[13.5,10*0.45*2],dpi=100)
#fig, axs = plt.subplots(2,3,figsize=[11*2./3*1.5,10*0.35*2],dpi=150)
fig, axs = plt.subplots(2,3,figsize=[11*2./3*1.5+1.25,10*0.35*2*0.9],dpi=150)
#axs = axs.tolist()
#plt.sca(axs[0])
lines11 = axs[0][0].plot(t, jnp.hstack((x0[:,:3],x0[:,3:]*s_)).dot(tobj.Ur),'-o',alpha=0.5,ms=2)
lines21 = [axs[0][0].plot([],[],'-o',lw=0.2,\
                  ms=4,markerfacecolor='None',markeredgecolor='black',alpha=0.5)[0] for _ in range(tobj.Ur.shape[1])]
for _1,_2 in zip(lines11,lines21):
    _2.set_markeredgecolor(_1.get_markeredgecolor())
    _2.set_color(_1.get_color())
#['A$_{NUM}$', 'B$_{NUM}$', 'C$_{NUM}$','A$_{NN}$','B$_{NN}$','C$_{NN}$']
#leg11 = axs[0][0].legend(iter(lines11+lines21), [_+__ for _ in pars[no]['sps'][:3] for __ in ['$_{NUM}$','$_{NN}$']],loc=7)
##axs[0][0].set_title('State Variables')
axs[0][0].set_xlabel('')
##axs[0][0].set_ylabel('C');
linesp1 = [axs[0][1].plot([],[],'o',lw=0.2,ms=3,alpha=0.4)[0] for _ in range(tobj.Ur.shape[1])]
leg21 = axs[0][1].legend(['$z^{{R}}_{{{}}}$'.format(i) for i in range(1,tobj.Ur.shape[1]+1)],loc='upper left',ncol=1, bbox_to_anchor=(0.005, 1.),labelspacing=0.3,columnspacing=0.3)#,loc=4,ncol=len(range(tobj.Ur.shape[1])),labelspacing=0.2,columnspacing=0.3)
axs[0][1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0][1].yaxis.set_major_locator(MaxNLocator(integer=True))
##axs[0][1].set_title('Whittened Derivatives Parity Plot')
axs[0][1].set_xlabel('')
##axs[0][1].set_ylabel('ANN');
##axs[0][2].set_title('Coupled Regression')
axs[0][2].set_xlabel('')
##axs[0][2].set_ylabel('Model Parameters')
lines31 = [axs[0][2].plot([], [],'.-',alpha=0.85,ms=7)[0] for _ in range(len(tobj.params[1][0][:nads]))]
#c = ['#d62728','#8c564b']
j = 0
for _ in range(len(lines31)):
    lines31[_].set_color('C'+str(j))
    if _%2==1:
        j+=1
#axs[0][2].plot([0,200],[1,1],'-',lw=0.5,alpha=0.2,color='blue')
#axs[0][2].legend(iter(lines31), ['$k_1$','$k_2$'],loc=4)
leg31 = axs[0][2].legend(iter(lines31), ['$k_{{{}}}$'.format(j+str(int(i)+1)) for i in range(int(len(tobj.params[1][0][:nads])/2.))  for j in ['','-']],loc='lower left',ncol=3, bbox_to_anchor=(-0.045, 1.005),labelspacing=0.2,columnspacing=0.3)
axs[0][2].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0][2].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[0][2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
lines31_ = axs[0][2].plot([],[],'-',lw=0.5,alpha=0.5,c='grey')

lines12 = axs[1][0].plot(t,  jnp.hstack((x0[:,:3],x0[:,3:]*s_)).dot(tobj.Un),'-o',alpha=0.5,ms=2)
lines22 = [axs[1][0].plot([],[],'-o',lw=0.2,\
                  ms=4,markerfacecolor='None',markeredgecolor='black',alpha=0.5)[0] for _ in range(tobj.Un.shape[1])]
for _1,_2 in zip(lines12,lines22):
    _2.set_markeredgecolor(_1.get_markeredgecolor())
    _2.set_color(_1.get_color())
#['A$_{NUM}$', 'B$_{NUM}$', 'C$_{NUM}$','A$_{NN}$','B$_{NN}$','C$_{NN}$']
#leg12 = axs[1][0].legend(iter(lines12+lines22), [_+__ for _ in pars[no]['sps'][3:] for __ in ['$_{NUM}$','$_{NN}$']],loc=7)
#axs[1][0].set_title('State Variables')
##axs[1][0].set_xlabel('Time')
##axs[1][0].set_ylabel('C');
linesp2 = [axs[1][1].plot([],[],'o',lw=0.2,ms=3,alpha=0.3)[0] for _ in range(tobj.Un.shape[1])]
leg22 = axs[1][1].legend(['$z^{{N}}_{{{}}}$'.format(i) for i in range(1,tobj.Un.shape[1]+1)],loc='upper left',ncol=1, bbox_to_anchor=(0.005, 1.),labelspacing=0.3,columnspacing=0.3)
axs[1][1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1][1].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[1][1].set_title('Whittened Derivatives Parity Plot')
##axs[1][1].set_xlabel('Model')
##axs[1][1].set_ylabel('ANN');
#axs[1][2].set_title('Coupled Regression')
##axs[1][2].set_xlabel('Ground Truth')
##axs[1][2].set_ylabel('Model Parameters')
lines32 = [axs[0][2].plot([], [],'.-',alpha=0.85,ms=7,mec='black')[0] for _ in range(len(tobj.params[1][0][nads:]))]
#c = ['#d62728','#8c564b']
j=6
for _ in range(len(lines32)):
    lines32[_].set_color('C'+str(j))
    if _%2==1:
        j+=1
#axs[1][2].plot([0,200],[1,1],'-',lw=0.5,alpha=0.2,color='blue')
axs[0][2].add_artist(leg31)
axs[0][2].legend(iter(lines32), ['$k_{{{}}}$'.format(j+str(int(nads/2)+i+1)) for i in range(int(len(tobj.params[1][0][nads:])/2.))  for j in ['','-']],loc='lower left',ncol=4, bbox_to_anchor=(.38, 1.005),labelspacing=0.2,columnspacing=0.3)
axs[1][2].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1][2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
lines32_ = axs[1][2].plot([],[],'-',lw=0.5,alpha=0.5,c='grey')


title = fig.suptitle('')
plt.tight_layout(rect=[0.,0.,1.,0.95])
#plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
plt.subplots_adjust(left, bottom, right, 0.9, wspace, hspace)

epochs = [0,1,2]
while epochs[-1]<len(opt_inv)-1:
    epochs += [int((epochs[-1]+1)**1.025)]
epochs[-1] = len(opt_inv[-1])-1
#pack = list(zip(epochs,[opt_inv[-1][int(_)] for _ in epochs]))
pack = list(zip(epochs,[[opt_inv[int(_)][1][0][0],opt_inv[int(_)][1][0][-1]] for _ in epochs]))
#pack = [[opt_inv[int(_)][0], [opt_inv[int(_)][1][0][n],opt_inv[int(_)][1][0][-1]]] for _ in epochs]

def init():
    l = 10
    axs[0][0].set_ylim([-0.455,0.805]) 
    axs[0][0].set_xlim([-0.025,1.025])
    axs[0][1].set_ylim([-l,l])
    axs[0][1].set_xlim([-l,l])
    axs[1][0].set_ylim([-0.025,0.525])
    axs[1][0].set_xlim([-0.025,1.025])
    axs[1][1].set_ylim([-l,l])
    axs[1][1].set_xlim([-l,l])
    axs[0][1].plot([-l,l],[-l,l],'-',lw=0.5,alpha=0.2,c='grey')
    axs[1][1].plot([-l,l],[-l,l],'-',lw=0.5,alpha=0.2,c='grey')
    return lines21+linesp1+lines31+lines22+linesp2+lines32#+axs

def animate(i):
    epoch, params = pack[i]
    nn_params, model_params = params
    title.set_text('epoch {:03d}'.format(int(epoch)))
    d = tobj.nn[n].batched_state(nn_params[n], t)
    for _ in range(tobj.Ur.shape[1]):
        lines21[_].set_data(t,d.dot(tobj.Ur)[:,_])
    for _ in range(tobj.Un.shape[1]):
        lines22[_].set_data(t,d.dot(tobj.Un)[:,_])
    x_parity = tobj.model.batched_eval([jnp.log(pars[4]['kinpars'])],[t,jnp.hstack((x0[:,:tobj.nobs],x0[:,tobj.nobs:]*s_))])#[:,:,0]
    y_parity = tobj.model.batched_eval(model_params,[t,tobj.nn[n].batched_state(nn_params[n],t)])#[:,:,0]
    x_parity, y_parity = [(_-_.mean(axis=0))/_.std(axis=0) for _ in [x_parity, y_parity]]
    for _ in range(tobj.Ur.shape[1]):
        linesp1[_].set_data(x_parity.dot(tobj.Ur)[:,_],y_parity.dot(tobj.Ur)[:,_])
    for _ in range(tobj.Un.shape[1]):
        linesp2[_].set_data(x_parity.dot(tobj.Un)[:,_],y_parity.dot(tobj.Un)[:,_])
    for _ in range(len(tobj.params[1][0][:nads])):
        lines31[_].set_data(jnp.log(pars[4]['kinpars'])[_],opt_inv[epoch][1][0][-1][0][_])
    for _ in range(nads,nads+len(tobj.params[1][0][nads:])):
        lines32[_-nads].set_data(jnp.log(pars[4]['kinpars'])[_],opt_inv[epoch][1][0][-1][0][_])
    #axs[0][2].set_xlim([-1,epoch])
#     axs[0][2].set_xlim([1.1*jnp.min(jnp.array([jnp.min(jnp.array(_.get_data()[0])) for _ in lines31])),
#                      1.1*jnp.max(jnp.array([jnp.max(jnp.array(_.get_data()[0])) for _ in lines31]))])
#     axs[0][2].set_ylim([1.1*jnp.min(jnp.array([jnp.min(jnp.array(_.get_data()[1])) for _ in lines31])),
#                      1.1*jnp.max(jnp.array([jnp.max(jnp.array(_.get_data()[1])) for _ in lines31]))])
#     #axs[1][2].set_xlim([-1,epoch])
#     axs[1][2].set_xlim([1.1*jnp.min(jnp.array([jnp.min(jnp.array(_.get_data()[0])) for _ in lines32])),
#                      1.1*jnp.max(jnp.array([jnp.max(jnp.array(_.get_data()[0])) for _ in lines32]))])
#     axs[1][2].set_ylim([0.9*jnp.min(jnp.array([jnp.min(jnp.array(_.get_data()[1])) for _ in lines32])),
#                      1.1*jnp.max(jnp.array([jnp.max(jnp.array(_.get_data()[1])) for _ in lines32]))])
    min31 = jnp.min(jnp.array([jnp.min(jnp.array([_.get_data()[0],_.get_data()[1]])) for l in [lines31,lines32] for _ in l]))
    max31 = jnp.max(jnp.array([jnp.max(jnp.array([_.get_data()[0],_.get_data()[1]])) for l in [lines31,lines32] for _ in l]))
    bds1 = [1.1*min31 if min31<0 else 0.9*min31,1.1*max31 if max31>0 else 0.9*max31]
    axs[0][2].set_xlim(bds1)
    axs[0][2].set_ylim(bds1)
    lines31_[0].set_data(bds1,bds1)
    min32 = jnp.min(jnp.array([jnp.min(jnp.array([_.get_data()[0],_.get_data()[1]])) for _ in lines32]))
    max32 = jnp.max(jnp.array([jnp.max(jnp.array([_.get_data()[0],_.get_data()[1]])) for _ in lines32]))
    bds2 = [1.1*min32 if min32<0 else 0.9*min32,1.1*max32 if max32>0 else 0.9*max32]
    axs[1][2].set_xlim(bds2)
    axs[1][2].set_ylim(bds2)
    lines32_[0].set_data(bds2,bds2)
    
    #dots.set_offsets( u[:,[0,i]])
    #line.set_data(u[:,0],u[:,i])
    return lines21+linesp1+lines31+lines22+linesp2+lines32+lines31_+lines32_#+axs

# Call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(epochs), interval=125, blit=True)

plt.close(anim._fig)

# Call function to display the animation
HTML(anim.to_html5_video())

anim.save('kin4rn_.gif', writer='imagemagick')

from IPython.display import Image
with open('kin4rn.gif','rb') as file:
    display(Image(file.read()))
    
# print('True parameters:      {}'.format(['{:.5f}'.format(float(_)) for _ in model_params0[0]]))
# print('Regressed parameters: {}'.format(['{:.5f}'.format(float(_)) for _ in model_params_inv[0]]))
