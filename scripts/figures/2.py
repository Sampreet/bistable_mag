# dependencies
from tqdm.rich import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl
import os
import sys

# quantrl modules
from quantrl.solvers.measure import QCMSolver

# add path to local libraries
sys.path.append(os.path.abspath(os.path.join('.')))
# import system
from envs.Optomagnomechanical import OMM_02_Vec

i = 1
dim_0 = 501
dim_1 = 501
dim = dim_1 * dim_0
libraries = ['jax', 'numpy']
methods = ['dopri5', 'vode']
X, Y = np.meshgrid(
    np.linspace(0.0, 0.2, dim_0),
    np.linspace(-2.0, 0.5, dim_1)
)

# environment
env = OMM_02_Vec(
    params={
        'P_d_SI'    : X.flatten(),
        'Delta_m'   : Y.flatten(),
    },
    branch=-1,
    t_norm_max=1000.0,
    t_norm_ssz=0.1,
    n_envs=dim,
    backend_library=libraries[i],
    action_interval=10,
    ode_method=methods[i],
    dir_prefix='data/omm_02/figures/2'
)

# magnon number
_, is_real_root, is_stable_root = env.get_N_ms()
sum_real_roots = np.sum(is_real_root, axis=1)
sum_stable_roots = np.sum(is_stable_root, axis=1)
Z = np.zeros(env.n_envs, dtype=int)
Z[sum_stable_roots == 3] = 3
Z[np.logical_and(sum_real_roots == 1, sum_stable_roots == 2)] = 2
Z[np.logical_and(sum_real_roots == 3, sum_stable_roots == 2)] = 1
Z = Z.reshape((dim_1, dim_0))
env.close(save=False)

# plot
plt.figure(figsize=(5.7, 5.7))
plt.grid(False)
plt.pcolormesh(X / 1e-3, Y, Z, cmap='nipy_spectral_r', vmin=0.0, vmax=3.0)
plt.xlabel('$P_{d}$ (mW)')
plt.ylabel('$\\Delta_{m} / \\omega_{b}$')
cbar = plt.colorbar(orientation='horizontal', location='top')
cbar.ax.set_xticks([0, 1, 2, 3], ['1S2U', '2S1U', '0S1U', '1S0U'])

# show
plt.tight_layout()
plt.show()
plt.close()