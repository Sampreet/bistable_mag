# dependencies
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.linalg as sl
import sys

# quantrl modules
from quantrl.solvers.measure import QCMSolver

# add path to local libraries
sys.path.append(os.path.abspath(os.path.join('.')))
# import system
from envs.Optomagnomechanical import OMM_02_Vec

i = 1
libraries = ['jax', 'numpy']
methods = ['dopri5', 'vode']
n_envs = 2001
T_SIs = np.logspace(-3, 1, n_envs)

# generate data
E_Ns = []
all_data = []
for branch in [0, -1]:
    # environment
    env = OMM_02_Vec(
        params={
            'T_SI': T_SIs
        },
        branch=branch,
        t_norm_max=100.0,
        t_norm_ssz=0.01,
        t_norm_mul=2.0 * np.pi,
        n_envs=n_envs,
        backend_library=libraries[i],
        action_interval=1000,
        cache_all_data=False,
        data_idxs=[0, -2],
        ode_method=methods[i],
        dir_prefix=f'data/omm_02/figures/r_1c'
    )
    env.evolve(close=False)
    np.savez_compressed(env.file_path_prefix + f'_data_{branch}', env.data)
    all_data.append(np.load(env.file_path_prefix + f'_data_{branch}.npz')['arr_0'])
    Modes = env.get_modes_steady_state()
    env.reset_states()
    As = env.get_A(None, Modes, None)
    Ds = env.D
    Corrs = np.zeros((n_envs, 6, 6), dtype=np.float_)
    for j in range(n_envs):
        Corrs[j] = sl.solve_lyapunov(As[j], - Ds[j])
    E_N = QCMSolver(
        Modes=None,
        Corrs=Corrs,
        params={
            'measure_codes': ['entan_ln_2'],
            'indices': (0, 1)
        }
    ).get_measures()[:, 0]
    E_Ns.append(E_N)
    env.close(save=False)

# figure
fig = plt.figure(figsize=(4.8, 4.8))
ax_0 = plt.gca()
ax_0.plot(T_SIs, E_Ns[0], 'b')
ax_0.plot(T_SIs, E_Ns[1], 'r')
ax_0.set_ylim(-0.02, 0.42)
ax_0.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
ax_0.set_yticks([0.025 * i for i in range(16)], minor=True)
ax_0.set_ylabel('$E_{am}$')
ax_0.set_xscale('log')
ax_0.set_xlabel('$T$ (mK)')
ax_0.legend(['lower', 'upper'], loc='upper right', frameon=False, borderpad=0.1, labelspacing=0.2, handletextpad=0.4, handlelength=1.8)
ax_0.grid(False)

# show
plt.tight_layout()
plt.show()
plt.close()