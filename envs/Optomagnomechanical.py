# dependencies
from tqdm.rich import tqdm
import numpy as np
import scipy.constants as sc

# quantrl modules
from quantrl.envs.deterministic import LinearizedHOEnv, LinearizedHOVecEnv
from quantrl.solvers.measure import QCMSolver

class OMM_02(LinearizedHOEnv):
    default_params = {
        'omega_a'       : 1e3,
        'omega_b'       : 1.0,
        'omega_b_SI'    : 2.0 * np.pi * 10e6,   # 2 * pi * 10 MegaHertz
        'kappa_a'       : 0.1,
        'kappa_m'       : 0.1,
        'kappa_b'       : 1e2 / 10e6,
        'Delta_a'       : - 0.9,
        'Delta_m'       : - 0.8,
        'K'             : 6.5e-9 / 10e6,
        'g_ma'          : 0.32,
        'g_mb'          : 1e-3 / 10e6,
        'P_d_SI'        : 50e-3,                # 50 milliWatts
        'T_SI'          : 10e-3                 # 10 milliKelvin
    }

    def __init__(self,
        params={},
        branch=2,
        t_norm_max=100.0,
        t_norm_ssz=0.1,
        t_norm_mul=1.0,
        action_interval=10,
        cache_all_data=False,
        data_idxs=[-2],
        backend_library='jax',
        ode_method='dopri5',
        dir_prefix='data/omm_02_vec'
    ):
        super().__init__(
            name='OMM_02_Vec',
            desc="Bistable Optomagnomechancial System",
            params=params,
            num_modes=3,
            num_quads=6,
            t_norm_max=t_norm_max,
            t_norm_ssz=t_norm_ssz,
            t_norm_mul=t_norm_mul,
            n_observations=42,
            n_properties=1,
            n_actions=1,
            action_maximums=[0.0],
            action_interval=action_interval,
            cache_all_data=cache_all_data,
            data_idxs=data_idxs,
            backend_library=backend_library,
            observation_space_range=[-1e300, 1e300],
            ode_method=ode_method,
            ode_atol=1e-12,
            ode_rtol=1e-9,
            plot=False,
            dir_prefix=dir_prefix
        )

        # set attributes
        self.branch = branch
        self.is_D_constant = True

        # update drift matrix
        indices_0 = [0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 5, 5]
        indices_1 = [0, 1, 3, 0, 1, 2, 1, 0, 4, 5, 4, 5]
        values = self.backend.convert_to_typed(
            tensor=[
                - self.params['kappa_a'],
                self.params['Delta_a'],
                self.params['g_ma'],
                - self.params['Delta_a'],
                - self.params['kappa_a'],
                - self.params['g_ma'],
                self.params['g_ma'],
                - self.params['g_ma'],
                - self.params['kappa_b'],
                self.params['omega_b'],
                - self.params['omega_b'],
                - self.params['kappa_b'],
            ],
            dtype='real'
        )
        self.A = self.backend.update(
            tensor=self.A,
            indices=(indices_0, indices_1),
            values=values
        )
    
    def reset_states(self):
        # initial phonon numbers
        n_th_0 = 0.0 if self.params['T_SI'] == 0 else 1 / (np.exp(sc.hbar * self.params['omega_b_SI'] / sc.k / self.params['T_SI']) - 1)

        # update noise matrix
        D = np.zeros(self.dim_corrs, dtype=np.float_)
        D[0, 0] = self.params['kappa_a']
        D[1, 1] = self.params['kappa_a']
        D[2, 2] = self.params['kappa_m']
        D[3, 3] = self.params['kappa_m']
        D[4, 4] = self.params['kappa_b'] * (2.0 * n_th_0 + 1.0)
        D[5, 5] = self.params['kappa_b'] * (2.0 * n_th_0 + 1.0)
        self.D = self.backend.convert_to_typed(
            tensor=D,
            dtype='real'
        )

        # initial values of the modes
        iv_modes = self.get_modes_steady_state()

        # initial values of the correlations
        iv_corrs = np.zeros(self.dim_corrs, dtype=np.float_)
        iv_corrs[0, 0] = 0.5
        iv_corrs[1, 1] = 0.5
        iv_corrs[2, 2] = 0.5
        iv_corrs[3, 3] = 0.5
        iv_corrs[4, 4] = n_th_0 + 0.5
        iv_corrs[5, 5] = n_th_0 + 0.5
        
        return self.backend.concatenate(
            tensors=(
                self.backend.real(iv_modes),
                self.backend.imag(iv_modes),
                self.backend.reshape(
                    tensor=iv_corrs,
                    shape=(self.num_corrs, )
                )
            ),
            axis=0,
            out=None
        )
    
    def get_N_ms(self):
        # controls
        P_d_SI = self.params['P_d_SI']
        omega_d = self.params['omega_a'] - self.params['Delta_a']
        Omega = np.sqrt(2.0 * self.params['kappa_m'] * P_d_SI / sc.hbar / omega_d) / self.params['omega_b_SI']

        # coeffiients
        eta = self.params['g_ma']**2 / (self.params['Delta_a']**2 + self.params['kappa_a']**2)
        zeta = self.params['g_mb']**2 / (self.params['omega_b']**2 + self.params['kappa_b']**2)
        kappa_0 = self.params['kappa_m'] + eta * self.params['kappa_a']
        Delta_0 = self.params['Delta_m'] - eta * self.params['Delta_a']
        K_p = 2.0 * (self.params['K'] - zeta * self.params['omega_b'])
        Coeffs = np.array([
            K_p**2,
            2.0 * Delta_0 * K_p,
            Delta_0**2 + kappa_0**2,
            - Omega**2
        ], dtype=np.float_)
        N_ms = np.zeros((3, ), dtype=np.float_)
        roots = np.roots(Coeffs)
        N_ms[np.imag(roots) == 0] = np.real(roots[np.imag(roots) == 0])

        is_real_root = np.zeros((3, ), dtype=int)
        is_stable_root = np.ones((3, ), dtype=int)
        is_real_root[np.imag(roots) == 0] = 1
        Modes = []
        for i in range(3):
            N_m = N_ms[i]
            # modes
            b = - self.params['g_mb'] * N_m / (self.params['omega_b'] - 1.0j * self.params['kappa_b'])
            Delta_m_p = self.params['Delta_m'] + 2.0 * self.params['K'] * N_m + 2.0 * self.params['g_mb'] * np.real(b)
            deno_a = self.params['Delta_a'] - 1.0j * self.params['kappa_a']
            deno_m = Delta_m_p - 1.0j * self.params['kappa_m']
            m = 0.0 if N_m == 0.0 else - 1.0j * Omega * deno_a / (deno_a * deno_m - self.params['g_ma']**2)
            a = - self.params['g_ma'] * m / deno_m

            modes = np.array([a, m, b], dtype=np.complex_)
            Modes.append(modes)
            A = self.get_A(modes, None, None)
            eigs, _ = np.linalg.eig(A)
            max_real_eig = np.max(np.real(eigs))
            is_stable_root[i] = 0 if max_real_eig > 0.0 else 1

        # select branch
        idxs = np.argsort(N_ms[is_real_root == 1]).tolist()

        return np.take(N_ms, idxs), np.take(is_stable_root, idxs)

    def get_modes_steady_state(self):
        N_m = self.get_N_ms()[0][self.branch]

        # controls
        P_d_SI = self.params['P_d_SI']
        omega_d = self.params['omega_a'] - self.params['Delta_a']
        Omega = np.sqrt(2.0 * self.params['kappa_m'] * P_d_SI / sc.hbar / omega_d) / self.params['omega_b_SI']

        # modes
        b = - self.params['g_mb'] * N_m / (self.params['omega_b'] - 1.0j * self.params['kappa_b'])
        Delta_m_p = self.params['Delta_m'] + 2.0 * self.params['K'] * N_m + 2.0 * self.params['g_mb'] * np.real(b)
        deno_a = self.params['Delta_a'] - 1.0j * self.params['kappa_a']
        deno_m = Delta_m_p - 1.0j * self.params['kappa_m']
        m = 0.0 if N_m == 0.0 else - 1.0j * Omega * deno_a / (deno_a * deno_m - self.params['g_ma']**2)
        a = - self.params['g_ma'] * m / deno_m

        return np.array([a, m, b], dtype=np.complex_)

    def get_A(self, t, modes, args):
        # modes
        a, m, b = modes

        # effective values
        G_mb = 2.0 * self.params['g_mb'] * m
        Delta_k = 2.0 * self.params['K'] * m**2
        Delta_m_p = self.params['Delta_m'] + 2.0 * self.params['K'] * self.backend.real(self.backend.conj(m) * m) + 2.0 * self.params['g_mb'] * self.backend.real(b) 
        Delta_m_pp = Delta_m_p + 2.0 * self.params['K'] * self.backend.real(self.backend.conj(m) * m)

        # update drift matrix
        indices_0 = [2, 2, 2, 3, 3, 3, 5, 5]
        indices_1 = [2, 3, 4, 2, 3, 4, 2, 3]
        values = [
            - self.params['kappa_m'] + self.backend.imag(Delta_k),
            Delta_m_pp - self.backend.real(Delta_k),
            self.backend.imag(G_mb),
            - Delta_m_pp - self.backend.real(Delta_k),
            - self.params['kappa_m'] - self.backend.imag(Delta_k),
            - self.backend.real(G_mb),
            - self.backend.real(G_mb),
            - self.backend.imag(G_mb)
        ]

        A = self.backend.update(
            tensor=self.A,
            indices=(indices_0, indices_1),
            values=values
        )
        del values

        return A

    def get_mode_rates(self, t, modes, args):
        # controls
        P_d_SI = self.params['P_d_SI']

        # effective paramaters
        omega_d = self.params['omega_a'] - self.params['Delta_a']
        Omega = self.backend.sqrt(2.0 * self.params['kappa_m'] * P_d_SI / sc.hbar / omega_d) / self.params['omega_b_SI']

        # modes
        a, m, b = self.backend.transpose(modes, 0, 1)

        # rates
        da_dt = - (1.0j * self.params['Delta_a'] + self.params['kappa_a']) * a - 1.0j * self.params['g_ma'] * m
        dm_dt = - (1.0j * self.params['Delta_m'] + self.params['kappa_m']) * m - 1.0j * self.params['g_ma'] * a - 2.0j * self.params['K'] * self.backend.conj(m) * m**2 - 1.0j * self.params['g_mb'] * m * (self.backend.conj(b) + b) + Omega
        db_dt = - (1.0j * self.params['omega_b'] + self.params['kappa_b']) * b - 1.0j * self.params['g_mb'] * self.backend.conj(m) * m

        return self.backend.stack(
            tensors=(da_dt, dm_dt, db_dt),
            axis=1,
            out=None
        )
    
    def get_properties(self):
        # extract modes and correlations
        Corrs = self.backend.convert_to_numpy(
            tensor=self.Observations[:, :, 2 * self.num_modes:],
            dtype='real'
        ).reshape(((self.action_interval + 1) * self.n_envs, *self.dim_corrs))

        # return properties
        return QCMSolver(
            Modes=None,
            Corrs=Corrs,
            params={
                'measure_codes': ['entan_ln_2'],
                'indices': (0, 1)
            }
        ).get_measures().reshape((self.action_interval + 1, self.n_envs, 1))
    
    def get_reward(self):
        return self.Properties[:, :, 0]

class OMM_02_Vec(LinearizedHOVecEnv):
    default_params = {
        'omega_a'       : 1e3,
        'omega_b'       : 1.0,
        'omega_b_SI'    : 2.0 * np.pi * 10e6,   # 2 * pi * 10 MegaHertz
        'kappa_a'       : 0.1,
        'kappa_m'       : 0.1,
        'kappa_b'       : 1e2 / 10e6,
        'Delta_a'       : - 0.9,
        'Delta_m'       : - 0.8,
        'K'             : 6.5e-9 / 10e6,
        'g_ma'          : 0.32,
        'g_mb'          : 1e-3 / 10e6,
        'P_d_SI'        : 50e-3,                # 50 milliWatts
        'T_SI'          : 10e-3                 # 10 milliKelvin
    }

    def __init__(self,
        params={},
        branch=-1,
        t_norm_max=100.0,
        t_norm_ssz=0.1,
        t_norm_mul=1.0,
        n_envs=101,
        action_interval=10,
        cache_all_data=False,
        data_idxs=[-2],
        backend_library='jax',
        ode_method='dopri5',
        dir_prefix='data/omm_02_vec'
    ):
        super().__init__(
            name='OMM_02_Vec',
            desc="Bistable Optomagnomechancial System",
            params=params,
            num_modes=3,
            num_quads=6,
            t_norm_max=t_norm_max,
            t_norm_ssz=t_norm_ssz,
            t_norm_mul=t_norm_mul,
            n_envs=n_envs,
            n_observations=42,
            n_properties=1,
            n_actions=1,
            action_maximums=[0.0],
            action_interval=action_interval,
            cache_all_data=cache_all_data,
            data_idxs=data_idxs,
            backend_library=backend_library,
            observation_space_range=[-1e300, 1e300],
            ode_method=ode_method,
            ode_atol=1e-12,
            ode_rtol=1e-9,
            plot=False,
            dir_prefix=dir_prefix
        )

        # set attributes
        self.branch = branch
        self.is_D_constant = True

        # update drift matrix
        indices_0 = [0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 5, 5]
        indices_1 = [0, 1, 3, 0, 1, 2, 1, 0, 4, 5, 4, 5]
        values = self.backend.convert_to_typed(
            tensor=[
                - self.params['kappa_a'],
                self.params['Delta_a'],
                self.params['g_ma'],
                - self.params['Delta_a'],
                - self.params['kappa_a'],
                - self.params['g_ma'],
                self.params['g_ma'],
                - self.params['g_ma'],
                - self.params['kappa_b'],
                self.params['omega_b'],
                - self.params['omega_b'],
                - self.params['kappa_b'],
            ],
            dtype='real'
        )
        self.A = self.backend.update(
            tensor=self.A,
            indices=(slice(None), indices_0, indices_1),
            values=self.backend.repeat(
                tensor=self.backend.reshape(
                    tensor=values,
                    shape=(1, len(indices_0))
                ),
                repeats=self.n_envs,
                axis=0
            )
        )
    
    def reset_states(self):
        # initial phonon numbers
        n_th_0 = 0.0 if isinstance(self.params['T_SI'], float) and self.params['T_SI'] == 0 else 1 / (np.exp(sc.hbar * self.params['omega_b_SI'] / sc.k / self.params['T_SI']) - 1)

        # update noise matrix
        D = np.zeros((self.n_envs, *self.dim_corrs), dtype=np.float_)
        D[:, 0, 0] = self.params['kappa_a']
        D[:, 1, 1] = self.params['kappa_a']
        D[:, 2, 2] = self.params['kappa_m']
        D[:, 3, 3] = self.params['kappa_m']
        D[:, 4, 4] = self.params['kappa_b'] * (2.0 * n_th_0 + 1.0)
        D[:, 5, 5] = self.params['kappa_b'] * (2.0 * n_th_0 + 1.0)
        self.D = self.backend.convert_to_typed(
            tensor=D,
            dtype='real'
        )

        # initial values of the modes
        iv_modes = self.get_modes_steady_state()

        # initial values of the correlations
        iv_corrs = np.zeros((self.n_envs, *self.dim_corrs), dtype=np.float_)
        iv_corrs[:, 0, 0] = 0.5
        iv_corrs[:, 1, 1] = 0.5
        iv_corrs[:, 2, 2] = 0.5
        iv_corrs[:, 3, 3] = 0.5
        iv_corrs[:, 4, 4] = n_th_0 + 0.5
        iv_corrs[:, 5, 5] = n_th_0 + 0.5
        
        return self.backend.concatenate(
            tensors=(
                self.backend.real(iv_modes),
                self.backend.imag(iv_modes),
                self.backend.convert_to_typed(
                    tensor=iv_corrs.reshape((self.n_envs, self.num_corrs)),
                    dtype='real'
                )
            ),
            axis=1,
            out=None
        )
    
    def get_N_ms(self):
        # controls
        P_d_SI = self.params['P_d_SI']
        omega_d = self.params['omega_a'] - self.params['Delta_a']
        Omega = self.backend.sqrt(2.0 * self.params['kappa_m'] * P_d_SI / sc.hbar / omega_d) / self.params['omega_b_SI']

        # coeffiients
        eta = self.params['g_ma']**2 / (self.params['Delta_a']**2 + self.params['kappa_a']**2)
        zeta = self.params['g_mb']**2 / (self.params['omega_b']**2 + self.params['kappa_b']**2)
        kappa_0 = self.params['kappa_m'] + eta * self.params['kappa_a']
        Delta_0 = self.params['Delta_m'] - eta * self.params['Delta_a']
        K_p = 2.0 * (self.params['K'] - zeta * self.params['omega_b'])
        Coeffs = [
            K_p**2,
            2.0 * Delta_0 * K_p,
            Delta_0**2 + kappa_0**2,
            - Omega**2
        ]
        for i in range(4):
            if len(np.shape(Coeffs[i])) == 0:
                Coeffs[i] = self.backend.convert_to_typed(
                    tensor=[Coeffs[i]] * self.n_envs,
                    dtype='real'
                )
        Coeffs = self.backend.convert_to_numpy(
            tensor=Coeffs,
            dtype='real'
        )

        is_real_root = np.zeros((self.n_envs, 3), dtype=int)
        is_stable_root = np.ones((self.n_envs, 3), dtype=int)
        N_ms = np.zeros((self.n_envs, 3), dtype=np.float_)
        for j in tqdm(
            range(self.n_envs),
            desc='Obtaining Steady States: ',
            leave=False,
            disable=False,
            mininterval=0.5
        ):
            roots = np.roots(Coeffs[:, j])
            is_real_root[j, np.imag(roots) == 0] = 1
            N_ms[j, np.imag(roots) == 0] = np.real(roots[np.imag(roots) == 0])

        for i in tqdm(
            range(3),
            desc='Checking Stability: ',
            leave=False,
            disable=False,
            mininterval=0.5
        ):
            N_m = N_ms[:, i]
            # modes
            b = - self.params['g_mb'] * N_m / (self.params['omega_b'] - 1.0j * self.params['kappa_b'])
            Delta_m_p = self.params['Delta_m'] + 2.0 * self.params['K'] * N_m + 2.0 * self.params['g_mb'] * self.backend.real(b)
            deno_a = self.params['Delta_a'] - 1.0j * self.params['kappa_a']
            deno_m = Delta_m_p - 1.0j * self.params['kappa_m']
            m = - 1.0j * Omega * deno_a / (deno_a * deno_m - self.params['g_ma']**2)
            m = self.backend.update(
                tensor=m,
                indices=(N_m == 0.0, ),
                values=0.0
            )
            a = - self.params['g_ma'] * m / deno_m

            modes = self.backend.stack(
                tensors=(a, m, b),
                axis=1,
                out=None
            )
            A = self.backend.convert_to_numpy(
                tensor=self.get_A(None, modes, None),
                dtype='real'
            )
            eigs, _ = np.linalg.eig(A)
            max_real_eig = np.max(np.real(eigs), axis=1)
            is_stable_root[max_real_eig > 0.0, i] = 0

        idxs = np.argsort(N_ms, axis=1)

        return (
            np.take_along_axis(N_ms, idxs, axis=1),
            np.take_along_axis(is_real_root, idxs, axis=1),
            np.take_along_axis(is_stable_root, idxs, axis=1)
        )

    def get_modes_steady_state(self):
        # controls
        P_d_SI = self.params['P_d_SI']
        omega_d = self.params['omega_a'] - self.params['Delta_a']
        Omega = self.backend.sqrt(2.0 * self.params['kappa_m'] * P_d_SI / sc.hbar / omega_d) / self.params['omega_b_SI']

        # magnon number
        _zero_arr = np.array([0])
        N_ms, is_real_root, is_stable_root = self.get_N_ms()
        sum_real_roots = np.sum(is_real_root, axis=1)
        N_m = np.array([N_ms[j, is_real_root[j] == 1][_zero_arr if sum_real_roots[j] == 1 else is_stable_root[j] == 1][self.branch] for j in range(self.n_envs)])

        # modes
        b = - self.params['g_mb'] * N_m / (self.params['omega_b'] - 1.0j * self.params['kappa_b'])
        Delta_m_p = self.params['Delta_m'] + 2.0 * self.params['K'] * N_m + 2.0 * self.params['g_mb'] * self.backend.real(b)
        deno_a = self.params['Delta_a'] - 1.0j * self.params['kappa_a']
        deno_m = Delta_m_p - 1.0j * self.params['kappa_m']
        m = - 1.0j * Omega * deno_a / (deno_a * deno_m - self.params['g_ma']**2)
        m = self.backend.update(
            tensor=m,
            indices=(N_m == 0.0, ),
            values=0.0
        )
        a = - self.params['g_ma'] * m / deno_m

        return self.backend.stack(
            tensors=(a, m, b),
            axis=1,
            out=None
        )

    def get_A(self, t, modes, args):
        # modes
        a, m, b = self.backend.transpose(modes)

        # effective values
        G_mb = 2.0 * self.params['g_mb'] * m
        Delta_k = 2.0 * self.params['K'] * m**2
        Delta_m_p = self.params['Delta_m'] + 2.0 * self.params['K'] * self.backend.real(self.backend.conj(m) * m) + 2.0 * self.params['g_mb'] * self.backend.real(b) 
        Delta_m_pp = Delta_m_p + 2.0 * self.params['K'] * self.backend.real(self.backend.conj(m) * m)

        # update drift matrix
        indices_0 = [2, 2, 2, 3, 3, 3, 5, 5]
        indices_1 = [2, 3, 4, 2, 3, 4, 2, 3]
        values = self.backend.stack(
            tensors=(
                - self.params['kappa_m'] + self.backend.imag(Delta_k),
                Delta_m_pp - self.backend.real(Delta_k),
                self.backend.imag(G_mb),
                - Delta_m_pp - self.backend.real(Delta_k),
                - self.params['kappa_m'] - self.backend.imag(Delta_k),
                - self.backend.real(G_mb),
                - self.backend.real(G_mb),
                - self.backend.imag(G_mb)
            ),
            axis=1,
            out=None
        )

        A = self.backend.update(
            tensor=self.A,
            indices=(slice(None), indices_0, indices_1),
            values=values
        )
        del values

        return A

    def get_mode_rates(self, t, modes, args):
        # controls
        P_d_SI = self.params['P_d_SI']

        # effective paramaters
        omega_d = self.params['omega_a'] - self.params['Delta_a']
        Omega = self.backend.sqrt(2.0 * self.params['kappa_m'] * P_d_SI / sc.hbar / omega_d) / self.params['omega_b_SI']

        # modes
        a, m, b = self.backend.transpose(modes, 0, 1)

        # rates
        da_dt = - (1.0j * self.params['Delta_a'] + self.params['kappa_a']) * a - 1.0j * self.params['g_ma'] * m
        dm_dt = - (1.0j * self.params['Delta_m'] + self.params['kappa_m']) * m - 1.0j * self.params['g_ma'] * a - 2.0j * self.params['K'] * self.backend.conj(m) * m**2 - 1.0j * self.params['g_mb'] * m * (self.backend.conj(b) + b) + Omega
        db_dt = - (1.0j * self.params['omega_b'] + self.params['kappa_b']) * b - 1.0j * self.params['g_mb'] * self.backend.conj(m) * m

        return self.backend.stack(
            tensors=(da_dt, dm_dt, db_dt),
            axis=1,
            out=None
        )
    
    def get_properties(self):
        # extract modes and correlations
        Corrs = self.backend.convert_to_numpy(
            tensor=self.Observations[:, :, 2 * self.num_modes:],
            dtype='real'
        ).reshape(((self.action_interval + 1) * self.n_envs, *self.dim_corrs))

        # return properties
        return QCMSolver(
            Modes=None,
            Corrs=Corrs,
            params={
                'measure_codes': ['entan_ln_2'],
                'indices': (0, 1)
            }
        ).get_measures().reshape((self.action_interval + 1, self.n_envs, 1))
    
    def get_reward(self):
        return self.Properties[:, :, 0]