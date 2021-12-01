#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# Created on Wed Jul+ 2020
#
# @author:gusmaogabriels@gmail.com // gusmaogabriels@gatech.edu
#
# ---

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

from kinn.basis.model import model
from kinn.basis.nnx import nn
from kinn.basis import plt, jit, optimizers, np, jnp, random
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# Graphics Aesthetics

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

# #### IC-Constrained Neural Net
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
# $$A\underset{k_2}{\stackrel{k_1}{\rightleftharpoons}} A*\notag\\
# B\underset{k_4}{\stackrel{k_3}{\rightleftharpoons}} B*\notag\\
# C\underset{k_6}{\stackrel{k_5}{\rightleftharpoons}} C*\notag\\
# A*+*\underset{k_8}{\stackrel{k_7}{\rightleftharpoons}} 2D*\notag\\
# B*+*\underset{k_10}{\stackrel{k_9}{\rightleftharpoons}} 2E*\notag\\
# D*\:+\:E*\underset{k_{12}}{\stackrel{k_{11}}{\rightleftharpoons}} F*\:+\:*\notag\\
# F*\:+\:E*\underset{k_{14}}{\stackrel{k_{13}}{\rightleftharpoons}} C*\:+\:*\notag$$

# Concentrations of chemical species $A$, $B$ and $C$ are observable variables, i.e. for topics [1], [2] and [3], there are not any latent (hidden) variables.

# A dictionary of stoichiometry matrices and with kinetic parameters (rate constants).
#
# All reactions are considered reversible

# ##### Parameter `dict` for trainer object

from trainer_source import pars

# Model parameters (rate constants): `model_params0`  
# Boundary conditions (concentrations at $t=0$): `bc0`

# Model function, `x` are state variables, `t` the independet variable (e.g. time).

from trainer_source import nn_c, nn_cn, nn_cn_bc, normtrig, nn_combo 
from trainer_source import trainer 

# ### Generate All Case Studios

# +
__reload__ = False
__errors__ = {}
__results__ = {}

addon = '_alpha13'
#_alphas = jnp.logspace(-2,4,5)
_alphas = jnp.logspace(-2,4,6)
for no in range(2,3):
    
    for tag, mode in zip(['fwd','inv','invsc','invvwn'],['forward']+['inverse']*3):
    
        if tag.startswith('inv'):
            alphas = _alphas
        else:
            alphas = [1]

        model_scale = 1e-2 # model scale (assume low so as not to be bias)
        model_nn = model(pars[no]['stoich'],model_scale=model_scale)
        if tag == 'fwd':
            model_nn.params = [pars[no]['kijnpars']]

        model_ = model(pars[no]['stoich'])
        model_.params = [pars[no]['kijnpars']]

        def gen_data(n_points,i):
            @jit
            def ode(t,C):
                return model_.single_eval([pars[no]['kijnpars']],[t,C]).flatten()

            tmax = 20 # max time to evaluate
            t_eval = (jnp.logspace(0,jnp.log10(tmax+1),n_points)-1.)/tmax
            print((pars[no]['bc'][i][0], tmax), pars[no]['bc'][i][1])
            sol = solve_ivp(ode, (pars[no]['bc'][i][0], tmax), pars[no]['bc'][i][1], t_eval = t_eval, method='LSODA',atol=1e-20,rtol=1e-20)

            return t_eval.reshape(-1,1), sol.y.T

        nnms = []
        nnts = []
        nncs = [] # one per dataset
        for i in range(len(pars[no]['bc'])):
            nnm = nn(**pars[no]['nnmpars'][mode])
            nnms += [nnm]
            nnt = nn(**pars[no]['nntpars'])
            nnts += [nnt]
            if mode == 'forward':
                nncs += [nn_combo([nnm,nnt], mode=mode,**pars[no]['nncpars'][i][mode])]
            elif mode == 'inverse': 
                nncs += [nn_combo([nnm], mode=mode,**pars[no]['nncpars'][i][mode])]
            else:
                raise Exception('mode not implemented ({})'.format(mode))

        num_epochs = 1000
        num_iter = 100

        if tag.endswith('sc') or tag.endswith('wn'):
            scale = True
        else:
            scale = False

        print('scale',scale)
        # create trainer object
        trainerf = trainer(nncs, model_nn, num_iter=num_iter, num_epochs=num_epochs, batch_size=1.,\
                               split=1., verbose=True, mode=mode, scale=scale, historian=True, tol=1e-10, nobs=pars[no]['nncpars'][i][mode]['nobs'],iter_data=[])
        trainerf.__setattr__('err_tags',['MODEL','DATA'])
        
        # reload to start from previous training
        if __reload__:# and not (tag.endswith('sc') or tag.endswith('wn')):
            try:
                trainerf.load('trainer_{}_{}{}.jnpz'.format(tag,no,addon))
            except:
                print('Load failed: {} {}'.format(tag,no))
                __errors__.update({'LOAD_FAILED':(tag,no)})
       
        # raw generated forward data
        datas0 = []
        for i in range(len(pars[no]['bc'])):
            t, x = gen_data(100,i)
            data0 = [(t,x.copy())]
            datas0 += data0
        dstack = jnp.vstack([_[1][:,pars[no]['nncpars'][i][mode]['nobs']:] for _ in datas0])
        s = jnp.std(dstack,axis=0)   

        # data generation
        datas  = []

        for i in range(len(pars[no]['bc'])):

            t, x = datas0[i]

            if tag == 'fwd': 
                data = [(t,[])]
            elif tag == 'inv':
                xinv  = x[:,:pars[no]['nncpars'][i][mode]['nobs']]
                data = [(t,xinv)]
            elif tag == 'invsc':
                d = x[:,pars[no]['nncpars'][i][mode]['nobs']:]
                d = d/s
                x_sc = jnp.hstack((x[:,:pars[no]['nncpars'][i][mode]['nobs']],d))
                data = [(t,x_sc)]
            elif tag == 'invvwn':
                d = x[:,pars[no]['nncpars'][i][mode]['nobs']:]
                d = d/s
                x_sc = jnp.hstack((x[:,:pars[no]['nncpars'][i][mode]['nobs']],d))
                x_scwn = x_sc + random.normal(random.PRNGKey(0),x_sc.shape)*0.025
                data = [(t,x_scwn)]
            datas  += data

        # setting optimizers
        for tol, h in zip([1e-12],[1e-3]): 
            trainerf.tol = tol  
            adam_kwargs  = {'step_size':h,'b1':0.9,'b2':0.9999,'eps':1e-12}
            trainerf.set_optimizer(optimizers.adam,adam_kwargs)
            trainerf.rejit()
            # train over ranges of alpha
            for alpha in alphas:
                trainerf.reinit()
                print(no,alpha)
                trainerf.train(datas,alpha=alpha)
        # refine solution with smaller step size
        for tol, h in zip([1e-12]*2,[1e-4,1e-5]): 
            trainerf.tol = tol  
            adam_kwargs  = {'step_size':h,'b1':0.9,'b2':0.9999,'eps':1e-12}
            trainerf.set_optimizer(optimizers.adam,adam_kwargs)
            trainerf.rejit()
            trainerf.reinit()
            print(no,alpha)
            trainerf.train(datas,alpha=alpha)
        
        # populate results dictionary
        __results__['trainer_{}_{}{}'.format(tag,no,addon)] = trainerf.get_error_metrics(datas0)

        # dump results into file
        trainerf.dump('trainer_{}_{}{}'.format(tag,no,addon))

# -

# ### Generate Pareto Sensitivities

# +
__reload__ = True
__errors__ = {}
__results__ = {}

addon = '_alpha17sens'
_alphas = jnp.concatenate((jnp.logspace(-6,6,21),jnp.logspace(-6,6,41)[-2::-1]))

for no in [3]:
    
    for tag, mode in zip(['invvwn','invsc'],['inverse']*2):
    
        if tag.startswith('inv'):
            alphas = _alphas
        else:
            alphas = [1]

        model_scale = 1e-1 # model scale (assume low so as not to be bias)
        model_nn = model(pars[no]['stoich'],model_scale=model_scale)
        if tag == 'fwd':
            model_nn.params = [pars[no]['kijnpars']]

        model_ = model(pars[no]['stoich'])
        model_.params = [pars[no]['kijnpars']]

        def gen_data(n_points,i):
            @jit
            def ode(t,C):
                return model_.single_eval([pars[no]['kijnpars']],[t,C]).flatten()

            tmax = 20 # max time to evaluate
            t_eval = (jnp.logspace(0,jnp.log10(tmax+1),n_points)-1.)/tmax
            print((pars[no]['bc'][i][0], tmax), pars[no]['bc'][i][1])
            sol = solve_ivp(ode, (pars[no]['bc'][i][0], tmax), pars[no]['bc'][i][1], t_eval = t_eval, method='LSODA',atol=1e-20,rtol=1e-20)
            return t_eval.reshape(-1,1), sol.y.T

        nnms = []
        nnts = []
        nncs = [] # one per dataset
        for i in range(len(pars[no]['bc'])):
            nnm = nn(**pars[no]['nnmpars'][mode])
            nnms += [nnm]
            nnt = nn(**pars[no]['nntpars'])
            nnts += [nnt]
            if mode == 'forward':
                nncs += [nn_combo([nnm,nnt], mode=mode,**pars[no]['nncpars'][i][mode])]
            elif mode == 'inverse': 
                nncs += [nn_combo([nnm], mode=mode,**pars[no]['nncpars'][i][mode])]
            else:
                raise Exception('mode not implemented ({})'.format(mode))

        num_epochs = 300
        num_iter   = 150

        if tag.endswith('sc') or tag.endswith('wn'):
            scale = True
        else:
            scale = False

        print('scale',scale)
        # create trainer object
        trainerf = trainer(nncs, model_nn, num_iter=num_iter, num_epochs=num_epochs, batch_size=1.,\
                               split=1., verbose=True, mode=mode, scale=scale, historian=True, tol=1e-10, nobs=pars[no]['nncpars'][i][mode]['nobs'],iter_data=[])
        trainerf.__setattr__('err_tags',['MODEL','DATA'])

        # reload to start from previous training
        if __reload__:# and not (tag.endswith('sc') or tag.endswith('wn')):
            try:
                trainerf.load('trainer_{}_{}{}.jnpz'.format(tag,no,addon))
            except:
                print('Load failed: {} {}'.format(tag,no))
                __errors__.update({'LOAD_FAILED':(tag,no)})

        # raw generated forward data
        datas0 = []
        for i in range(len(pars[no]['bc'])):
            t, x = gen_data(100,i)
            data0 = [(t,x.copy())]
            datas0 += data0
        dstack = jnp.vstack([_[1][:,pars[no]['nncpars'][i][mode]['nobs']:] for _ in datas0])
        s = jnp.std(dstack,axis=0)

        datas  = []
        
        # data generation
        for i in range(len(pars[no]['bc'])):

            t, x = datas0[i]

            if tag == 'fwd': 
                data = [(t,[])]
            elif tag == 'inv':
                xinv  = x[:,:pars[no]['nncpars'][i][mode]['nobs']]
                data = [(t,xinv)]
            elif tag == 'invsc':
                d = x[:,pars[no]['nncpars'][i][mode]['nobs']:]
                d = d/s
                x_sc = jnp.hstack((x[:,:pars[no]['nncpars'][i][mode]['nobs']],d))
                data = [(t,x_sc)]
            elif tag == 'invvwn':
                d = x[:,pars[no]['nncpars'][i][mode]['nobs']:]
                d = d/s
                x_sc = jnp.hstack((x[:,:pars[no]['nncpars'][i][mode]['nobs']],d))
                x_scwn = x_sc + random.normal(random.PRNGKey(0),x_sc.shape)*0.025
                data = [(t,x_scwn)]
            datas  += data
        
        # setting optimizers
        trainerf.tol = 1e-12  
        adam_kwargs  = {'step_size':1e-4,'b1':0.9,'b2':0.9999,'eps':1e-12}
        trainerf.set_optimizer(optimizers.adam,adam_kwargs)
        trainerf.rejit()
            
        # train over ranges of alpha
        for alpha in alphas:
            trainerf.alpha=alpha
            trainerf.reinit()
            print(no,alpha)
            trainerf.train(datas,alpha=alpha)

        # create results dictionary
        __results__['trainer_{}_{}{}'.format(tag,no,addon)] = trainerf.get_error_metrics(datas0)

        # dump results into file
        trainerf.dump('trainer_{}_{}{}'.format(tag,no,addon))
