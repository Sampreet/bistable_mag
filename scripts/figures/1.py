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
        data_idxs=[0, 3, 6],
        ode_method=methods[i],
        dir_prefix='data/omm_02/figures/1'
    )
    env.evolve(close=False)
    np.savez_compressed(env.file_path_prefix + f'_data_{branch}', env.data)
    all_data.append(np.load(env.file_path_prefix + f'_data_{branch}.npz')['arr_0'])
    N_ms, is_real_root, is_stable_root = env.get_N_ms()
    env.close(save=False)

# data
xs = np.linspace(90, 100, 11)
ts = np.linspace(90, 100, 1001)
ms = np.array([data[:, :, 1] + 1.0j * data[:, :, 2] for data in all_data])
N_m_ts = np.real(np.conjugate(ms) * ms)
N_m_avgs = np.mean(N_m_ts[..., -972:], axis=-1)
N_ms[is_real_root == 0] = np.NaN
is_real_stable_root = np.zeros((n_envs, 3), dtype=int)
is_real_stable_root[np.logical_and(is_real_root == 1, is_stable_root == 1)] = 1
num_stable_roots = np.sum(is_real_stable_root, axis=1)
colors = np.ones((n_envs, 3), dtype=int) * 9
colors[is_stable_root == 1] = 1

# figure
fig = plt.figure(figsize=(4.8, 4.8))
ax_0 = plt.gca()
l_0 = ax_0.plot([0, 1], [np.inf, np.inf], 'b', linestyle='--', linewidth=1.5)[0]
l_1 = ax_0.plot([0, 1], [np.inf, np.inf], 'y', linestyle='--', linewidth=1.5)[0]
l_2 = ax_0.plot([0, 1], [np.inf, np.inf], 'r', linestyle='--', linewidth=1.5)[0]
l_3 = ax_0.scatter([0, 1], [np.inf, np.inf], c='k', s=20, marker='s')
l_4 = ax_0.scatter([0, 1], [np.inf, np.inf], c='k', s=15, marker='D')
l_5 = ax_0.scatter([0, 1], [np.inf, np.inf], c='k', s=25, marker='o')
ax_0.plot([13.84, 13.84], [-10, 20], color='k', linestyle=':', linewidth=0.75)
ax_0.plot([81.94, 81.94], [-10, 20], color='k', linestyle=':', linewidth=0.75)
ax_0.plot([126.34, 126.34], [-10, 20], color='k', linestyle=':', linewidth=0.75)
for j in range(3):
    ax_0.scatter(P_d_SIs * 1000, N_ms[:, j] / 1e14, s=2, c=['b' if c == 1 else ('y' if j == 1 else 'r') for c in colors[:, j]])
# ax_0.scatter(P_d_SIs[100:801:100] * 1000, N_m_avgs[0, 100:801:100] / 1e14, c='k', s=20, marker='s')
ax_0.scatter([50], [N_m_ts[0, 500, 0] / 1e14], c='k', s=50, marker='s')
# ax_0.scatter(P_d_SIs[200:1201:100] * 1000, N_m_avgs[-1, 100:1201:100] / 1e14, c='k', s=15, marker='D')
ax_0.scatter([50], [N_m_ts[-1, 500, 0] / 1e14], c='k', s=50, marker='D')
# ax_0.scatter(P_d_SIs[1300::100] * 1000, N_m_avgs[-1, 1300::100] / 1e14, c='k', s=25, marker='o')
ax_0.scatter([130], [N_m_ts[-1, 1300, 0] / 1e14], c='k', s=50, marker='o')
ax_0.set_ylim(-0.5, 10.5)
ax_0.set_ylabel('$10^{-14} \\times I$')
ax_0.set_xlim(-10, 210)
ax_0.set_xlabel('$P_{d}$ (mW)')
ax_0.fill_betweenx([-0.5, 10.5], [13.84, 13.84], [81.94, 81.94], color='k', alpha=0.05)
ax_0.fill_betweenx([-0.5, 10.5], [126.3, 126.3], [250.0, 250.0], color='r', alpha=0.05)
ax_0.legend([l_0, (l_1, l_2), (l_3, l_4, l_5)], ['stable', 'unstable', 'avg. dyna.'], loc='upper right', frameon=False, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, borderpad=0.1, labelspacing=0.2, handletextpad=0.4, handlelength=1.8)
ax_0.grid(False)

# inset 1
ax_0.plot([49, 112], [0.64, 3.18], color='k', linestyle='--', linewidth=0.75)
ax_0.plot([51, 198], [0.56, 1.68], color='k', linestyle='--', linewidth=0.75)
ax_1 = fig.add_axes([0.6, 0.325, 0.3, 0.1])
ax_1.plot(ts, N_m_ts[0, 500, -1001::1] / 1e14, color='k', linestyle='-', linewidth=0.75)
ax_1.plot(xs, [N_m_ts[0, 500, 0] / 1e14] * len(xs), color='b', linestyle='--', linewidth=1.5)
ax_1.scatter(xs, [N_m_avgs[0, 500] / 1e14] * len(xs), c='k', s=20, marker='s')
ax_1.fill_betweenx([-0.5, 10.5], [90, 90], [100, 100], color='k', alpha=0.05)
ax_1.set_xlabel('$t / \\tau$', labelpad=0)
ax_1.set_xlim(90, 100)
ax_1.set_ylim(0.6, 0.7)
ax_1.grid(False)

# inset 2
ax_0.plot([49, 9], [6.49, 8.5], color='k', linestyle='--', linewidth=0.75)
ax_0.plot([51, 97], [6.49, 8.5], color='k', linestyle='--', linewidth=0.75)
ax_2 = fig.add_axes([0.25, 0.8, 0.3, 0.1])
ax_2.plot(ts, N_m_ts[-1, 500, -1001::1] / 1e14, color='k', linestyle='-', linewidth=0.75)
ax_2.plot(xs, [N_m_ts[-1, 500, 0] / 1e14] * len(xs), color='b', linestyle='--', linewidth=1.5)
ax_2.scatter(xs, [N_m_avgs[-1, 500] / 1e14] * len(xs), c='k', s=15, marker='D')
ax_2.fill_betweenx([-0.5, 10.5], [90, 90], [100, 100], color='k', alpha=0.05)
ax_2.set_xlabel('$t / \\tau$', labelpad=0)
ax_2.set_xlim(90, 100)
ax_2.set_ylim(6, 7)
ax_2.grid(False)

# inset 3
ax_0.plot([129, 112], [7.36, 6.4], color='k', linestyle='--', linewidth=0.75)
ax_0.plot([131, 198], [7.36, 6.4], color='k', linestyle='--', linewidth=0.75)
ax_3 = fig.add_axes([0.6, 0.5, 0.3, 0.15])
ax_3.plot(ts, N_m_ts[-1, 1300, -1001::1] / 1e14, color='k', linestyle='-', linewidth=1.0)
ax_3.plot(xs, [N_m_ts[-1, 1300, 0] / 1e14] * len(xs), color='r', linestyle='--', linewidth=1.5)
ax_3.scatter(xs, [N_m_avgs[-1, 1300] / 1e14] * len(xs), c='k', s=25, marker='o')
ax_3.fill_betweenx([-0.5, 10.5], [90, 90], [100, 100], color='r', alpha=0.05)
ax_3.set_xlim(90, 100)
ax_3.set_ylim(5, 10)
ax_3.grid(False)

# show
plt.tight_layout()
plt.show()
plt.close()