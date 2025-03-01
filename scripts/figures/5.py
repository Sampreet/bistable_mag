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
    dir_prefix='data/omm_02/figures/5'
)

# simulate
iv = env.reset_states()
modes = iv[:, :env.num_modes] + 1.0j * iv[:, env.num_modes:2 * env.num_modes]
A = env.get_A(None, modes, None)
D = env.D
Corrs = np.zeros((dim, 6, 6), dtype=np.float_)
for j in tqdm(
    range(dim),
    desc='Calculating Correlations: ',
    leave=False,
    disable=False,
    mininterval=0.5
):
    Corrs[j] = sl.solve_lyapunov(A[j], - D[j])
E_N = QCMSolver(
    Modes=None,
    Corrs=Corrs,
    params={
        'measure_codes': ['entan_ln_2'],
        'indices': (0, 1)
    }
).get_measures().reshape((dim_1, dim_0))
N_m = np.real(np.conjugate(modes[:, 1]) * modes[:, 1]).reshape((dim_1, dim_0))
E_N[N_m == 0.0] = np.NaN
np.savez_compressed(env.file_path_prefix + '_entan_ln_ss', E_N)
E_N = np.load(env.file_path_prefix + '_entan_ln_ss.npz')['arr_0']
env.close(save=False)

# plot
plt.figure()
plt.grid(False)
plt.pcolormesh(X / 1e-3, Y, E_N, cmap='nipy_spectral_r', vmin=0.0, vmax=0.6)
plt.xlabel('$P_{d}$ (mW)')
plt.ylabel('$\\Delta_{m} / \\omega_{b}$')
plt.colorbar().ax.set_title('$E_{N}^{(ss)}$')

# show
plt.tight_layout()
plt.show()
plt.close()