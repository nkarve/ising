# ising
## A simple Ising model simulation using Monte Carlo time stepping
### Features:
- Real-time interactive simulation of 2D classical Ising model on a square lattice using `matplotlib`
- Computation of macroscopic system variables: energy, magnetization, susceptibility, specific heat
- Support for single-flip Metropolis algorithm (thermalizes faster on average) and cluster-flip Wolff algorithm (avoids critical slowing down by reducing autocorrelation) 
- Usage of `numba` JIT compilation and `joblib` parallelization for extra speed

### Getting Started:

```
> git clone https://github.com/nkarve/ising.git
> cd src
> python ising_realtime.py
```

<img src="/demos/rt.gif">
