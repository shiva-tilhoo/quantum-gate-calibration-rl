import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# -----------------------------
# 1. Qutrit parameters
# -----------------------------

N = 3                  # dimension of the system (qutrit)
w = 5.0 * 2*np.pi      # "central" frequency (arbitrary units)
alpha = -0.3 * 2*np.pi # anharmonicity (non-equidistant levels, transmon-like)

# Annihilation (b) and creation (b†) operators
b = qt.destroy(N)

# Free Hamiltonian (Duffing-type anharmonic oscillator)
H0 = w * b.dag()*b + 0.5 * alpha * b.dag()*b.dag()*b*b

# -----------------------------
# 2. Drive (control pulse)
# -----------------------------
#
# A drive is applied on (b + b†) with a time-dependent amplitude: eps(t)
# The idea: simulate a microwave pulse that excites the qubit.

Omega0 = 0.1 * 2*np.pi  # max drive amplitude
T_pulse = 20.0          # total duration of the simulation
n_steps = 1000
tlist = np.linspace(0, T_pulse, n_steps)

def eps_t(t, args):
    """
    Centred Gaussian pulse.
    The shape can be changed (sine, square, etc.).
    """
    t0 = T_pulse / 2
    sigma = T_pulse / 6
    return Omega0 * np.exp(- (t - t0)**2 / (2*sigma**2))

# Drive Hamiltonian: H_drive(t) = eps(t) * (b + b†)
H_drive = [b + b.dag(), eps_t]

# Total time-dependent Hamiltonian
H = [H0, H_drive]

# -----------------------------
# 3. Decoherence (T1 / T2 noise)
# -----------------------------
#
# T1 : relaxation (|1> -> |0>, |2> -> |1>)
# T2: dephasing (loss of coherence) on |1> and |2>

# T1 and T2 as "times" (arbitrary units)
T1_10 = 50.0   # relaxation 1 -> 0
T1_21 = 40.0   # relaxation 2 -> 1
T2_1  = 1.0  # dephasing of level |1>
T2_2  = 25.0   # dephasing of level |2>

gamma_10 = 1.0 / T1_10
gamma_21 = 1.0 / T1_21
gamma_phi1 = 1.0 / T2_1
gamma_phi2 = 1.0 / T2_2

# Relaxation jump operators
e0 = qt.basis(N, 0)
e1 = qt.basis(N, 1)
e2 = qt.basis(N, 2)

L_10 = np.sqrt(gamma_10) * e0 * e1.dag()  # |1> -> |0>
L_21 = np.sqrt(gamma_21) * e1 * e2.dag()  # |2> -> |1>

# Dephasing operators (the trace is removed for pure dephasing)
P1 = e1 * e1.dag()
P2 = e2 * e2.dag()
L_phi1 = np.sqrt(gamma_phi1) * P1
L_phi2 = np.sqrt(gamma_phi2) * P2

c_ops = [L_10, L_21, L_phi1, L_phi2]

# -----------------------------
# 4. Initial state and observables
# -----------------------------

# Start from |0>
psi0 = e0
rho0 = psi0 * psi0.dag()

# Observables: projectors on |0>, |1>, |2>
P0 = e0 * e0.dag()
P1 = e1 * e1.dag()
P2 = e2 * e2.dag()

e_ops = [P0, P1, P2]

# -----------------------------
# 5. Simulation of the dynamics
# -----------------------------

result = qt.mesolve(H, rho0, tlist, c_ops, e_ops)

pop0 = np.array(result.expect[0])  # population of |0>
pop1 = np.array(result.expect[1])  # population of |1>
pop2 = np.array(result.expect[2])  # population of |2> (leakage)

# -----------------------------
# 6. Simple analysis: leakage & "fidelity"
# -----------------------------
#
# To implement, for instance, a complete X gate |0> -> |1>,
# the final fidelity can be approximated as: F = population of |1>
# (the phase is ignored in this first, simple study).

F_final = pop1[-1]
leak_final = pop2[-1]

print(f"Approximate fidelity (|0> -> |1>) at t = {T_pulse:.2f} : F = {F_final:.4f}")
print(f"Leakage to level |2> at t = {T_pulse:.2f} : L = {leak_final:.4f}")

# -----------------------------
# 7. Population plot
# -----------------------------

plt.figure()
plt.plot(tlist, pop0, label="P(|0>)")
plt.plot(tlist, pop1, label="P(|1>)")
plt.plot(tlist, pop2, label="P(|2>) (leakage)")
plt.xlabel("Time (arbitrary units)")
plt.ylabel("Population")
plt.legend()
plt.title("Qutrit dynamics with drive + decoherence")
plt.grid(True)
plt.show()
