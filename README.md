### KINNs: Kinetics-Informed Neural Networks
---
#### Pareto + Regularization Approach

arXiv: https://doi.org/10.48550/arXiv.2011.14473

***Exemplary KINNs training*** for the *dcs* reaction type

<p align="center">
  <img src="./misc/gifs/kinn4.gif" alt="animated" width="1000"/>
</p>

<img src="https://render.githubusercontent.com/render/math?math=\dot{\mathbf{x}}(t)"> is the automatic differentatiation of a the neural-network <img src="https://render.githubusercontent.com/render/math?math=\mathbf{x}(t)"> over time points <img src="https://render.githubusercontent.com/render/math?math=t">. The physical mode (microkinetic model) is denoted by <img src="https://render.githubusercontent.com/render/math?math=f_\mathbf{p}{(\cdot)}"> with parameters <img src="https://render.githubusercontent.com/render/math?math=\mathbf{p}=\ln(\mathbf{k})">.

The kinetic model represents the following fully-reversible chemical reactions.   
The latent Kinetics type *d* involves ad/desorption steps, intermediate species (radicals, <img src="https://render.githubusercontent.com/render/math?math=D^*">) that do not have a corresponding gas phase species is part of the reaction, type *c*.   
Reaction between radicals ![formula](https://render.githubusercontent.com/render/math?math=D^*), ![formula](https://render.githubusercontent.com/render/math?math=E^*), and ![formula](https://render.githubusercontent.com/render/math?math=F^*), adds further complexity to surface reaction, type *s*.   


<div align="center"><img src="https://render.githubusercontent.com/render/math?math=A%2B\*\underset{k_{-1}}{\stackrel{k_1}{\rightleftharpoons}}A*"></div>   
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=B\underset{k_{-2}}{\stackrel{k_2}{\rightleftharpoons}}B*"></div>   
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=C\underset{k_{-3}}{\stackrel{k_3}{\rightleftharpoons}}C*"></div>   
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=A*%2B*\underset{k_{-4}}{\stackrel{k_4}{\rightleftharpoons}}2D*"></div>   
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=B*%2B*\underset{k_{-5}}{\stackrel{k_5}{\rightleftharpoons}}2E*"></div>   
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=D*%2BE*\underset{k_{-6}}{\stackrel{k_{6}}{\rightleftharpoons}}F*\%2B*"></div>   
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=F*%2BE*\underset{k_{-7}}{\stackrel{k_{13}}{\rightleftharpoons}}C*\%2B*"></div>    

Raw preliminary [JAX](https://github.com/google/jax)-based source code can be found under [kinn](./kinn).

#### Reference Jupyter Notebooks

   1. Combined data generation  can be found [here](./paper/kinn_datagen_reg.ipynb).
   2. Data digestion and plot generation [here](./paper/kinn_plotsgen.ipynb).
