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
P_d_SIs = np.linspace(0.0, 0.2, n_envs)

# generate data
E_Ns = []
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
        data_idxs=[0, -2],
        ode_method=methods[i],
        dir_prefix=f'data/omm_02/figures/3'
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

# data
xs = np.linspace(90, 100, 11)
ts = np.linspace(90, 100, 1001)
E_N_ts = np.array([data[:, :, 1] for data in all_data])
E_N_avgs = np.mean(E_N_ts[-1, :, -972:], axis=1)

# figure
fig = plt.figure(figsize=(4.8, 4.8))
ax_0 = plt.gca()
l_0 = ax_0.plot([0, 1], [np.inf, np.inf], 'g', linestyle='--', linewidth=1.5)[0]
l_3 = ax_0.scatter([0, 1], [np.inf, np.inf], c='k', s=20, marker='s')
l_4 = ax_0.scatter([0, 1], [np.inf, np.inf], c='k', s=15, marker='D')
l_5 = ax_0.scatter([0, 1], [np.inf, np.inf], c='k', s=25, marker='o')
ax_0.plot([13.84, 13.84], [-10, 20], color='k', linestyle=':', linewidth=0.75)
ax_0.plot([81.94, 81.94], [-10, 20], color='k', linestyle=':', linewidth=0.75)
ax_0.plot([126.34, 126.34], [-10, 20], color='k', linestyle=':', linewidth=0.75)
for j in range(2):
    ax_0.scatter(P_d_SIs[:1264] * 1000, E_Ns[j][:1264], s=2, c='g')
ax_0.scatter(P_d_SIs[1300::140] * 1000, E_N_avgs[1300::140], c='k', s=25, marker='o')
ax_0.set_ylim(-0.04, 0.84)
ax_0.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax_0.set_yticks([0.05 * i for i in range(16)], minor=True)
ax_0.set_ylabel('$E_{am}$')
ax_0.set_xlim(-10, 210)
ax_0.set_xlabel('$P_{d}$ (mW)')
ax_0.fill_betweenx([-0.5, 1.0], [13.84, 13.84], [81.94, 81.94], color='k', alpha=0.05)
ax_0.fill_betweenx([-0.5, 1.0], [126.3, 126.3], [250.0, 250.0], color='r', alpha=0.05)
ax_0.legend([l_0, (l_3, l_4, l_5)], ['stable', 'avg. dyna.'], loc='upper left', frameon=False, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, borderpad=0.1, labelspacing=0.2, handletextpad=0.4, handlelength=1.8)
ax_0.grid(False)

# inset 1
ax_0.plot([50, 61], [0.03, 0.315], color='k', linestyle='--', linewidth=0.75)
ax_0.plot([50, 148], [0.02, 0.13], color='k', linestyle='--', linewidth=0.75)
ax_0.scatter([50], [E_Ns[0][500]], c='k', s=50, marker='s')
ax_1 = fig.add_axes([0.424, 0.32, 0.3, 0.16])
ax_1.plot(ts, E_N_ts[0, 500, -1001:], color='k', linestyle='-', linewidth=0.75)
ax_1.plot(xs, [E_Ns[0][500]] * len(xs), color='g', linestyle='--', linewidth=1.5)
ax_1.scatter(ts[::100], E_N_ts[0, 500, -1001::100], c='k', s=20, marker='s')
ax_1.plot(ts, E_N_ts[-1, 500, -1001:], color='k', linestyle='-', linewidth=0.75)
ax_1.plot(xs, [E_Ns[-1][500]] * len(xs), color='g', linestyle='--', linewidth=1.5)
ax_1.scatter(ts[::100], E_N_ts[-1, 500, -1001::100], c='k', s=15, marker='D')
ax_1.fill_betweenx([-0.5, 1.0], [90, 90], [100, 100], color='k', alpha=0.05)
ax_1.set_xlabel('$t / \\tau$', labelpad=0)
ax_1.set_xlim(90, 100)
ax_1.set_ylim(-0.1, 0.5)
ax_1.set_yticks([0.0, 0.4])
ax_1.grid(False)

# inset 2
ax_0.plot([49, 61], [0.375, 0.13], color='k', linestyle='--', linewidth=0.75)
ax_0.plot([51, 148], [0.375, 0.315], color='k', linestyle='--', linewidth=0.75)
ax_0.scatter([50], [E_Ns[-1][500]], c='k', s=50, marker='D')
# ax_2 = fig.add_axes([0.4, 0.42, 0.3, 0.08])
# ax_2.plot(xs, [E_Ns[-1][500]] * len(xs), color='g', linestyle='--', linewidth=1.5)
# ax_2.scatter(ts[::100], E_N_ts[-1, 500, -1001::100], c='k', s=15, marker='D')
# ax_2.fill_betweenx([-0.5, 1.0], [90, 90], [100, 100], color='k', alpha=0.05)
# ax_2.set_xlabel('$t / \\tau$', labelpad=0)
# ax_2.set_xlim(90, 100)
# ax_2.set_xticks([90, 95, 100], labels=['']* 3)
# ax_2.set_ylim(0.3, 0.4)
# ax_2.grid(False)

# inset 3
ax_0.plot([129, 112], [0.49, 0.606], color='k', linestyle='--', linewidth=0.75)
ax_0.plot([131, 198], [0.49, 0.606], color='k', linestyle='--', linewidth=0.75)
ax_0.scatter([130], [E_N_avgs[1300]], c='k', s=50, marker='o')
ax_3 = fig.add_axes([0.6, 0.74, 0.3, 0.16])
ax_3.plot(ts, E_N_ts[-1, 1300, -1001:], color='k', linestyle='-', linewidth=1.0)
ax_3.scatter(xs, [E_N_avgs[1300]] * len(xs), c='k', s=25, marker='o')
ax_3.fill_betweenx([-0.5, 1.0], [90, 90], [100, 100], color='r', alpha=0.05)
ax_3.set_xlabel('$t / \\tau$', labelpad=0)
ax_3.set_xlim(90, 100)
ax_3.set_ylim(0.4, 0.6)
ax_3.grid(False)

# show
plt.tight_layout()
plt.show()
plt.close()