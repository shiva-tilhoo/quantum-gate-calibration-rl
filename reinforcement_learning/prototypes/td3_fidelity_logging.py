import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv

from qutip_qtrl import pulseoptim as cpo

# -------- LaTex-style fonts (no external LaTeX needed).--------
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 120,
})

import gymnasium as gym
from gymnasium import spaces

from qutip import (
    Qobj,
    qeye,
    destroy,
    tensor,
)

# KEY CHANGE: import TD3 instead of SAC
# from stable_baselines3 import SAC
from stable_baselines3 import TD3 
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


# =========================
# Gates & fidelity
# =========================

def cz3_gate():
    """
    2-qutrit CZ gate:
    diag(1,1,1, 1, e^{2πi/3}, e^{-2πi/3}, 1, e^{-2πi/3}, e^{2πi/3})
    """
    phases = np.array([
        1,
        1,
        1,
        1,
        np.exp(2j*np.pi/3),
        np.exp(-2j*np.pi/3),
        1,
        np.exp(-2j*np.pi/3),
        np.exp(2j*np.pi/3),
    ], dtype=complex)

    return Qobj(np.diag(phases), dims=[[3, 3], [3, 3]])


def average_gate_fidelity(U, U_target):
    """F_avg(U, U_target) = (|Tr(U_t^† U)|^2 + d)/(d(d+1))."""
    d = U.shape[0]
    M = U_target.dag() * U
    trM = M.tr()
    F = (abs(trM)**2 + d) / (d * (d + 1))
    # clip to avoid F > 1 because of floating-point errors
    return float(np.clip(np.real(F), 0.0, 1.0))


# =========================
# System construction (2 qutrits)
# =========================

def build_two_qutrit_system(
    n_segments=160,
    dt=10.0,
    max_amp=0.3,
    w1=1.0,
    w2=0.9,
    chi1=-0.11,
    chi2=-0.11,
    g=0.0025,
):
    """
    Hamiltonian:
    H = sum_i[w_i a_i† a_i + χ_i a_i† a_i† a_i a_i + ε_i(t)(a_i+a_i†)]
        + g (a1 + a1†)(a2 + a2†)
    """

    n_levels = 3
    dim = n_levels ** 2

    a = destroy(n_levels)
    I = qeye(n_levels)
    a1 = tensor(a, I)
    a2 = tensor(I, a)

    n1 = a1.dag() * a1
    n2 = a2.dag() * a2

    H0 = (
        w1 * n1
        + w2 * n2
        + chi1 * (a1.dag() * a1.dag() * a1 * a1)
        + chi2 * (a2.dag() * a2.dag() * a2 * a2)
        + g * ((a1 + a1.dag()) * (a2 + a2.dag()))
    )

    Hc1 = (a1 + a1.dag())
    Hc2 = (a2 + a2.dag())

    Hd = Qobj(H0.full(), dims=[[dim], [dim]])
    Hc1_q = Qobj(Hc1.full(), dims=[[dim], [dim]])
    Hc2_q = Qobj(Hc2.full(), dims=[[dim], [dim]])

    U0 = Qobj(qeye(dim).full(), dims=[[dim], [dim]])
    U_targ = Qobj(cz3_gate().full(), dims=[[dim], [dim]])

    evo_time = n_segments * dt

    return {
        "Hd": Hd,
        "Hcs": [Hc1_q, Hc2_q],
        "U0": U0,
        "U_targ": U_targ,
        "n_segments": n_segments,
        "dt": dt,
        "evo_time": evo_time,
        "max_amp": max_amp,
        "dim": dim,
        "a1": a1,
        "a2": a2,
        "w1_nom": w1,
        "w2_nom": w2,
        "chi1_nom": chi1,
        "chi2_nom": chi2,
        "g_nom": g,
    }


def make_H0_tensored(a1, a2, w1, w2, chi1, chi2, g):
    n1 = a1.dag() * a1
    n2 = a2.dag() * a2
    H0 = (
        w1 * n1
        + w2 * n2
        + chi1 * (a1.dag() * a1.dag() * a1 * a1)
        + chi2 * (a2.dag() * a2.dag() * a2 * a2)
        + g * ((a1 + a1.dag()) * (a2 + a2.dag()))
    )
    return H0


def propagate_with_pulse(H0, Hc1, Hc2, eps1, eps2, dt):
    U = tensor(qeye(3), qeye(3))
    for j in range(len(eps1)):
        H = H0 + eps1[j] * Hc1 + eps2[j] * Hc2
        U_step = (-1j * H * dt).expm()
        U = U_step * U
    return U


def fidelity_trajectory(H0, Hc1, Hc2, eps1, eps2, dt):
    U = tensor(qeye(3), qeye(3))
    U_targ = cz3_gate()
    traj = []
    for j in range(len(eps1)):
        H = H0 + eps1[j] * Hc1 + eps2[j] * Hc2
        U_step = (-1j * H * dt).expm()
        U = U_step * U
        traj.append(average_gate_fidelity(U, U_targ))
    return np.array(traj)


# =========================
# OCT / GRAPE
# =========================

def run_oct_grape_two_qutrit_cz(
    n_segments=160,
    dt=10.0,
    max_amp=0.3,
    g=0.0025,
    n_iters=4000,
    fid_err_targ=1e-10,
):
    """
    Run OCT on the nominal Hamiltonian (no noise).
    Return pulses and fidelity history vs iteration.
    """

    sys = build_two_qutrit_system(
        n_segments=n_segments,
        dt=dt,
        max_amp=max_amp,
        g=g,
    )

    Hd = sys["Hd"]
    Hcs = sys["Hcs"]
    U0 = sys["U0"]
    U_targ = sys["U_targ"]
    evo_time = sys["evo_time"]

    print(f"Dim = {sys['dim']}, T = {evo_time}, n_segments = {n_segments}")

    result = cpo.optimize_pulse_unitary(
        H_d=Hd,
        H_c=Hcs,
        U_0=U0,
        U_targ=U_targ,
        num_tslots=n_segments,
        evo_time=evo_time,
        amp_lbound=-max_amp,
        amp_ubound=max_amp,
        fid_err_targ=fid_err_targ,
        max_iter=n_iters,
        init_pulse_type="RND",
        log_level=0,
        gen_stats=True,
    )

    fid_err = result.fid_err
    F_final = 1.0 - fid_err

    print("\nOCT / GRAPE finished (nominal):")
    print(f" final fid_err = {fid_err:.3e}")
    print(f" final fidelity (process overlap) = {F_final:.6f}")

    pulses = result.final_amps  # shape [n_ts, 2]
    eps1 = pulses[:, 0]
    eps2 = pulses[:, 1]

    # check fidelity with our own propagator
    a1 = sys["a1"]
    a2 = sys["a2"]
    w1_nom = sys["w1_nom"]
    w2_nom = sys["w2_nom"]
    chi1_nom = sys["chi1_nom"]
    chi2_nom = sys["chi2_nom"]
    g_nom = sys["g_nom"]

    H0_nom = make_H0_tensored(a1, a2, w1_nom, w2_nom, chi1_nom, chi2_nom, g_nom)
    Hc1_tens = (a1 + a1.dag())
    Hc2_tens = (a2 + a2.dag())

    U = propagate_with_pulse(H0_nom, Hc1_tens, Hc2_tens, eps1, eps2, dt)
    U_targ_tens = cz3_gate()
    F_check = average_gate_fidelity(U, U_targ_tens)

    print(f" average gate fidelity (our check, nominal) = {F_check:.6f}")

    sys["H0_nom"] = H0_nom
    sys["Hc1_tens"] = Hc1_tens
    sys["Hc2_tens"] = Hc2_tens

    fid_hist = None
    if hasattr(result, "stats") and hasattr(result.stats, "fid_err"):
        fid_hist = 1.0 - np.array(result.stats.fid_err)

    return sys, eps1, eps2, F_final, F_check, fid_hist


# =========================
# Bandit-style residual env
# =========================

class CZResidualBanditEnv(gym.Env):
    """
    One-step environment:

    - Observation: normalized nominal OCT pulses (fixed)
    - Action: residual corrections over all segments at once
    - Reward: F(U) - F_baseline on noisy device
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        eps1_base,
        eps2_base,
        sys_nominal,
        max_amp=0.3,
        max_delta=0.02,
        noise_std_w=0.001,
        noise_std_g=0.0005,
        seed=None
    ):

        super().__init__()

        self.eps1_base = np.array(eps1_base, dtype=float)
        self.eps2_base = np.array(eps2_base, dtype=float)
        self.n_segments = len(self.eps1_base)
        self.max_delta = max_delta
        self.max_amp = max_amp
        self.dt = sys_nominal["dt"]

        # nominal operators
        self.a1 = sys_nominal["a1"]
        self.a2 = sys_nominal["a2"]
        self.w1_nom = sys_nominal["w1_nom"]
        self.w2_nom = sys_nominal["w2_nom"]
        self.chi1_nom = sys_nominal["chi1_nom"]
        self.chi2_nom = sys_nominal["chi2_nom"]
        self.g_nom = sys_nominal["g_nom"]

        self.Hc1 = sys_nominal["Hc1_tens"]
        self.Hc2 = sys_nominal["Hc2_tens"]

        rng = np.random.default_rng(seed)

        # noisy device sample
        dw1 = rng.normal(0.0, noise_std_w)
        dw2 = rng.normal(0.0, noise_std_w)
        dg = rng.normal(0.0, noise_std_g)

        self.w1_eff = self.w1_nom + dw1
        self.w2_eff = self.w2_nom + dw2
        self.g_eff = self.g_nom + dg

        print(f"Fixed noisy device: w1={self.w1_eff:.4f}, w2={self.w2_eff:.4f}, g={self.g_eff:.6f}")

        self.H0_eff = make_H0_tensored(
            self.a1, self.a2,
            self.w1_eff, self.w2_eff,
            self.chi1_nom, self.chi2_nom, self.g_eff
        )

        self.U_target = cz3_gate()

        # baseline fidelity
        U_baseline = propagate_with_pulse(
            self.H0_eff, self.Hc1, self.Hc2,
            self.eps1_base, self.eps2_base,
            self.dt
        )

        self.F_baseline = average_gate_fidelity(U_baseline, self.U_target)
        print(f"Baseline OCT fidelity on noisy device: {self.F_baseline:.6f}")

        # observation
        base_obs = np.concatenate([self.eps1_base, self.eps2_base])
        max_val = np.max(np.abs(base_obs)) + 1e-8
        self.base_obs = (base_obs / max_val).astype(np.float32)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=self.base_obs.shape,
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 * self.n_segments,),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        return self.base_obs.copy(), {}

    def step(self, action):

        action = np.clip(action, -1.0, 1.0)
        delta = (self.max_delta * action).reshape(self.n_segments, 2)

        eps1 = self.eps1_base + delta[:, 0]
        eps2 = self.eps2_base + delta[:, 1]

        eps1 = np.clip(eps1, -self.max_amp, self.max_amp)
        eps2 = np.clip(eps2, -self.max_amp, self.max_amp)

        U = propagate_with_pulse(self.H0_eff, self.Hc1, self.Hc2, eps1, eps2, self.dt)
        F = average_gate_fidelity(U, self.U_target)

        reward = F - self.F_baseline

        info = {"fidelity": float(F)}
        return self.base_obs.copy(), float(reward), True, False, info

    def render(self): pass
    def close(self): pass


# =========================
# RL helpers
# =========================

class FidelityLoggingCallback(BaseCallback):
    """
    Callback that records the fidelity of each episode:
    - in memory (self.fidelities, self.episode_indices)
    - AND in a CSV file (save_path)
    """
    def __init__(self, verbose=0, save_path="fidelity_log.csv"):
        super().__init__(verbose)
        self.save_path = save_path
        self.fidelities = []
        self.episode_indices = []

        # Create the CSV file with its headers
        with open(self.save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "fidelity"])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                F = info.get("fidelity", None)
                if F is not None:
                    ep = len(self.fidelities)
                    self.fidelities.append(F)
                    self.episode_indices.append(ep)

                    # append to the CSV
                    with open(self.save_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([ep, F])

        return True


# =========================
# Main
# =========================

def main():
    # ---------- 1) OCT on nominal System --------
    n_segments = 160
    dt = 10.0
    max_amp = 0.3
    g = 0.0025

    n_iters = 4000
    fid_err_targ = 1e-10

    sys, eps1_oct, eps2_oct, F_proc_nom, F_avg_nom, fid_hist = run_oct_grape_two_qutrit_cz(
        n_segments=n_segments,
        dt=dt,
        max_amp=max_amp,
        g=g,
        n_iters=n_iters,
        fid_err_targ=fid_err_targ,
    )

    print(
        f"\nNominal OCT:"
        f" process fidelity = {F_proc_nom:.6f},"
        f" avg gate fidelity = {F_avg_nom:.6f}"
    )
    # ... plotting code for the OCT pulses and the fidelity history ...

    # -------- 2) Bandit env: fixed noisy device --------
    noise_std_w = 0.001
    noise_std_g = 0.00005

    env = CZResidualBanditEnv(
        eps1_base=eps1_oct,
        eps2_base=eps2_oct,
        sys_nominal=sys,
        max_amp=max_amp,
        max_delta=0.02,
        noise_std_w=noise_std_w,
        noise_std_g=noise_std_g,
        seed=123,
    )

    print(f"Baseline OCT fidelity on noisy device: {env.F_baseline:.6f}")

    # ------------ 3) TD3 training --------------
    n_actions = env.action_space.shape[0]
    
    # Exploration noise for TD3 (NormalActionNoise)
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.3 * np.ones(n_actions),
    )

    policy_kwargs = dict(net_arch=[256, 256, 256])
    fidelity_cb = FidelityLoggingCallback(save_path="fidelity_log.csv")

    model = TD3( # KEY CHANGE: use TD3
        "MlpPolicy",
        env,
        action_noise=action_noise, # TD3 uses the exploration noise explicitly
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        policy_delay=2, # Key TD3 hyperparameter (policy updated every 2 steps)
        verbose=1,
        tensorboard_log=None,
        policy_kwargs=policy_kwargs,
    )

    total_timesteps = 50_000
    model.learn(total_timesteps=total_timesteps, callback=fidelity_cb)
    model.save("two_qutrit_cz_td3_bandit_noise") # Updated model name

    # ... rest of the code for the plots and the final evaluation ...

    # ------------ 4) RL learning curve (scatter only) -------------
    raw_fids = np.array(fidelity_cb.fidelities)
    episodes = np.array(fidelity_cb.episode_indices)

    plt.figure(figsize=(7, 4))
    plt.scatter(episodes, raw_fids, s=8, alpha=0.6, label="RL episodes")
    plt.axhline(env.F_baseline, linestyle=":", linewidth=2,
                label="OCT baseline (noisy device)")
    plt.xlabel("Episode")
    plt.ylabel("Fidelity on noisy device")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rl_learning_scatter.png", dpi=300)
    plt.show()

    # --------- 5) Evaluate final RL policy ----------
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    _, reward, _, _, info = env.step(action)
    F_rl = info["fidelity"]

    delta = (env.max_delta * np.clip(action, -1.0, 1.0)).reshape(env.n_segments, 2)
    eps1_rl = np.clip(env.eps1_base + delta[:, 0], -env.max_amp, env.max_amp)
    eps2_rl = np.clip(env.eps2_base + delta[:, 1], -env.max_amp, env.max_amp)

    print(f"\nBaseline OCT fidelity on noisy device: {env.F_baseline:.6f}")
    print(f"Final RL pulse fidelity on same noisy device: {F_rl:.6f}")
    
    # ... plotting code for fidelity vs time and OCT vs RL pulses ...

if __name__ == "__main__":
    main()