# dependencies
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# add path to local libraries
sys.path.append(os.path.abspath(os.path.join('.')))
# import system
from envs.Optomagnomechanical import OMM_02_Vec

i = 1
libraries = ['jax', 'numpy']
methods = ['dopri5', 'vode']
n_envs = 2001
P_d_SIs = np.linspace(0.0, 0.2, n_envs)

# generate data
all_data = []
for branch in [0, -1]:
    # environment
    env = OMM_02_Vec(
        params={
            'P_d_SI': P_d_SIs
        },
        branch=branch,
        t_norm_max=100.0,
        t_norm_ssz=0.01,
        t_norm_mul=2.0 * np.pi,
        n_envs=n_envs,
        backend_library=libraries[i],
        action_interval=1000,
        cache_all_data=False,
        data_idxs=(np.arange(0, 36) + 8).tolist(),
        ode_method=methods[i],
        dir_prefix='data/omm_02/figures/4'
    )
    env.evolve(close=False)
    np.savez_compressed(env.file_path_prefix + f'_data_{branch}', env.data[:, 3000::3000])
    env.close(save=False)