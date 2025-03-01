from qom.solvers.measure import get_Wigner_distributions_single_mode, QCMSolver
import numpy as np
import matplotlib.pyplot as plt

branches = [0, -1, -1]
indices = [500, 500, 1300]
for j in range(3):
    Corrs = np.load(f'data/omm_02/figures/4/100.0_0.01_6.283185307179586_[0.0]_1000/lho_vec_env_data_{branches[j]}.npz')['arr_0'][indices[j]].reshape((3, 6, 6))

    wigners = get_Wigner_distributions_single_mode(Corrs, params={
        'indices'   : [1],
        'wigner_xs' : np.linspace(-10, 10, 1001),
        'wigner_ys' : np.linspace(-10, 10, 1001) 
    })[:, 0]
    E_Ns = QCMSolver(
        Modes=None,
        Corrs=Corrs,
        params={
            'measure_codes' : ['entan_ln_2'],
            'indices'       : (0, 1),
        }
    ).get_measures()[:, 0]
    for i in range(3):
        print(E_Ns[i])
        np.savetxt(f'data/omm_02/figures/4/wigner_{j}_{i}.txt', np.transpose(wigners[i]))