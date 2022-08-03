from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy.ndimage import convolve, generate_binary_structure
from scipy.stats import linregress 

plt.style.use('ggplot')
plt.rcParams['axes.grid'] = False

@numba.njit("(i4[:,:])(i4[:,:], f8)", nogil=True)
def update(grid, temp):
    ''' Performs a single time-step update on a grid using the Metropolis algorithm 
    
    Chooses L^2 random points and performs the MCMC step

    Parameters
    ----------
    grid: array
        The spin lattice of 0s and 1s
    temp: float
        The temperature of the system
    
    Returns
    _______
    array[:, :]
        The updated grid
    '''
    
    L = grid.shape[0]
    beta = 1. / temp
    
    xs = np.random.randint(0, L, size=(L, L))
    ys = np.random.randint(0, L, size=(L, L))

    for i in range(L):
        for j in range(L):
            x, y = xs[i, j], ys[i, j]      
            s = grid[x, y]
            neighbours = grid[(x+1)%L, y] + grid[(x-1)%L, y] + grid[x, (y+1)%L] + grid[x, (y-1)%L]
            cost = 2 * s * neighbours

            if cost < 0 or np.random.random() < np.exp(-cost * beta):
                grid[x, y] = -s

    return grid


''' Defines the energy and magnetization operators on the lattice '''

kernel = generate_binary_structure(2, 1)
kernel[1, 1] = False

def energy_kernel(grid):
    return -0.5 * np.sum(grid * convolve(grid, kernel, mode='constant', cval=0))

magnetization = np.sum


@numba.njit("UniTuple(f8[:], 2)(i4[:,:], f8, f8, i8, i8, boolean)", nogil=True)
def update_metropolis(grid, temp, energy, nupdates, nmeasures, periodic=True):
    ''' Performs multiple time-step updates on a grid using the Metropolis algorithm 
    
    Also samples the intensive energy and magnetization of the system during the final time-steps. 
    To save time, energy is computed incrementally instead of using the kernel.

    Parameters
    ----------
    grid: array
        The spin lattice of 0s and 1s
    temp: float
        The temperature of the system
    energy: float
        The initial energy of the system (accounting for heat bath)
    nupdates: int
        The number of time-steps to update the grid by
    nmeasures: int
        The number of time-steps at the end in which to sample energy and magnetization
    periodic: bool
        Whether the lattice is periodic in both directions (torus BCs)
    
    Returns
    _______
    tuple(list, list)
        energies array, magnetization array
    '''

    E = np.zeros(nmeasures)
    M = np.zeros(nmeasures)
    
    L = grid.shape[0]
    beta = 1. / temp

    for i in range(nupdates):
        x, y = np.random.randint(0, L), np.random.randint(0, L)      
        s = grid[x, y]

        if periodic:
            neighbours = grid[(x+1)%L, y] + grid[(x-1)%L, y] + grid[x, (y+1)%L] + grid[x, (y-1)%L]
            dE = 2 * s * neighbours
        else:
            l, r, u, d = 0, 0, 0, 0
            if x != 0:   l = grid[x-1, y]
            if x != L-1: r = grid[x+1, y]
            if y != 0:   u = grid[x, y-1]
            if y != L-1: d = grid[x, y+1]

            dE = 2 * s * (l + r + u + d)

        if dE < 0 or np.random.random() < np.exp(-dE * beta):
            grid[x, y] = -s
            energy += dE
        
        k = i - (nupdates - nmeasures)
        if k >= 0:
            E[k] = energy
            M[k] = magnetization(grid)
    
    E /= L * L 
    M /= L * L
    
    return E, M

@numba.njit("(i4[:,:])(i4[:,:], f8)", nogil=True)
def update_wolff(grid, temp):
    ''' Performs a single time-step update on a periodic grid using the Wolff algorithm 
    
    The Wolff algorithm flips clusters rather than individual spins to reduce autocorrelation and 
    hence increase sample accuracy.

    Parameters
    ----------
    grid: array
        The spin lattice of 0s and 1s
    temp: float
        The temperature of the system

    Returns
    _______
    array[:,:]
        The updated spin lattice
    '''
    
    L = grid.shape[0]
    prob = 1. - np.exp(-2. / temp)

    z0, z1 = np.random.randint(0, L, 2)
    stack = [(z0, z1)]
    spin = grid[z0, z1]
    state = np.zeros((L, L), dtype=np.bool_)

    while len(stack) > 0:
        x, y = stack.pop()
        nbrs = [(x,(y+1)%L), (x,(y-1)%L), ((x+1)%L,y), ((x-1)%L,y)]
        if np.random.random() < prob:
            for nx, ny in nbrs:
                if not state[nx][ny] and grid[nx, ny] == spin:
                    stack.append((nx, ny))
                    state[nx][ny] = True
                    grid[nx, ny] = -grid[nx, ny]
    return grid


def plot_macroscopic(temps, eqsteps, calcsteps, L):
    ''' Plots all macroscopic quantities of a system after it has reached equilibrium 
    
    Parameters
    ----------
    temps: list(float)
        The list of temperatures at which to plot
    eqsteps: int
        The number of timesteps until equilibrium is declared
    calcsteps: int
        The number of timesteps considered after equilibrium to obtain the mean macroscopic variables
    L: int
        The size of the spin lattice
    '''

    E = np.zeros_like(temps) # Energy
    M = np.zeros_like(temps) # Magnetization
    C = np.zeros_like(temps) # Specific heat capacity
    X = np.zeros_like(temps) # Magnetic susceptibility
    R = np.zeros_like(temps) # Correlation length
    
    def get_macros(i, temp):
        print(f'Running phase {(i+1)}\n', end='')
        grid = 2 * np.random.randint(0, 2, size=(L, L), dtype=int) - 1

        ''' To use the Wolff algorithm:

        e = np.zeros(calcsteps)
        m = np.zeros(calcsteps)

        for j in range(eqsteps):
            grid = update_wolff(grid, temp)

            k = j - (eqsteps - calcsteps)
            if k >= 0:
                e[k] = energy_kernel(grid)
                m[k] = magnetization(grid)

        e /= L * L
        m /= L * L

        '''
        e, m = update_metropolis(grid, temp, energy_kernel(grid), eqsteps, calcsteps, periodic=True)

        E[i] = np.mean(e)
        M[i] = np.mean(m)
        C[i] = np.var(e) / (temp * temp)
        X[i] = np.var(m) / temp

    Parallel(n_jobs=6, prefer='threads')(delayed(get_macros)(i, temp) for i, temp in enumerate(temps))
    
    print(E, M, C, X, sep='\n')

    plt.figure()
    plt.xlabel('Temperature')
    plt.ylabel('Energy')
    plt.scatter(temps, E)
    
    plt.figure()
    plt.xlabel('Temperature')
    plt.ylabel('Specific Heat')
    plt.scatter(temps, C)
    plt.axvline(x=2.236, color='black', ls='--')
    
    plt.figure()
    plt.xlabel('Temperature')
    plt.ylabel('Susceptibility')
    plt.scatter(temps, X)
    plt.axvline(x=2.236, color='black', ls='--')

    plt.figure()
    plt.xlabel('Temperature')
    plt.ylabel('Magnetisation')
    plt.scatter(temps, np.abs(M))
    plt.axvline(x=2.236, color='black', ls='--')

    
@numba.njit("(f8)(i4[:,:], i4)", nogil=True)
def correlation(grid, dist):
    '''Computes the correlation between spins at a given distance
    
    Parameters
    ----------
    grid: array
        The spin lattice of 0s and 1s
    dist: int
        The distance at which to measure correlation
    '''

    L = grid.shape[0]
    A = L * L
    SS = S0 = Sx = 0

    for i in range(L):
        for j in range(L):
            nbrs = grid[((i+dist)%L,j)] + grid[((i-dist)%L,j)] + grid[(i,(j+dist)%L)] + grid[(i,(j-dist)%L)]
            S0 += grid[i, j]
            Sx += nbrs
            SS += grid[i, j] * nbrs

    return SS / (4. * A) - S0 * Sx / (4 * A * A)


def correlation_length(grid, maxdist):
    '''Computes the correlation length for a given lattice
    
    Parameters
    ----------
    grid: array
        The spin lattice of 0s and 1s
    maxdist: int
        The maximum distance at which to measure correlation
    '''
    
    dist = np.arange(1, maxdist)
    corr = np.zeros_like(dist, dtype=np.float32)
    for i in dist:
        corr[i-1] = np.clip(correlation(grid, i), 1e-5, None)
    
    logcorr = -np.log(corr)
    res = linregress(dist, logcorr)
    return 1 / res.slope
