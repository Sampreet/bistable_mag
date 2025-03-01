# dependencies
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import numpy as np

# quantrl modules
from quantrl.plotters import TrajectoryPlotter

# load data
fig = plt.figure(figsize=(9.6, 9.0))
axes = fig.subplots(3, 3, width_ratios=[0.2, 0.2, 0.2], height_ratios=[0.2, 0.2, 0.2])
cmap = 'RdBu_r'
times = [30, 60, 90]
indices = [
    slice(250, 751),
    slice(250, 751),
    slice(None)
]
ticks = [
    [0, 250, 500],
    [0, 250, 500],
    [0, 500, 1000]
]
ticklabels = [
    ['-5', '0', '5'],
    ['-5', '0', '5'],
    ['-10', '0', '10']
]
maxs = [
    [4.0e-1, 4.0e-1, 4.0e-1],
    [1.2e-1, 1.2e-1, 1.2e-1],
    [1.2e-2, 7.2e-3, 6.0e-3],
]
powers = [
    [0, 0, 0],
    [-1, -1, -1],
    [-2, -3, -3],
]
multipliers = [
    [0.1, 0.1, 0.1],
    [0.3, 0.3, 0.3],
    [0.3, 1.8, 1.5],
]
wigners = []
for j in range(3):
    for i in range(3):
        print(j, i)
        ax = axes[j, i]
        wigner = np.loadtxt(f'data/omm_02/figures/4/wigner_{j}_{i}.txt')
        ax.pcolormesh(wigner[indices[j], indices[j]], vmin=0.0, vmax=maxs[j][i], cmap=cmap)
        ax.set_xticks(ticks[j], ticklabels[j])
        ax.set_xlabel('$\\delta X_{m}$')
        ax.set_yticks(ticks[j], ticklabels[j])
        ax.set_ylabel('$\\delta Y_{m}$', labelpad=-8 if j == 2 else 0)
        ax.set_title(f'$t = {times[i]} \\tau$' if j == 0 else '', pad=12)
        norm = Normalize(vmin=0.0, vmax=maxs[j][i])
        map = ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(map, ax=ax)
        cbar.ax.set_title('$10^{' + str(powers[j][i]) + '} \\times \\mathcal{W} (u_{m})$' if powers[j][i] != 0 else '$\\mathcal{W} (u_{m})$', size=12, pad=12)
        cbar.set_ticks([multipliers[j][i] * k * 10**powers[j][i] for k in range(5)], labels=[f'{multipliers[j][i] * k:0.1f}' for k in range(5)])

# show
plt.tight_layout()
plt.show()
plt.close()