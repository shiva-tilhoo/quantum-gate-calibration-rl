import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# -----------------------------
# 1. Qubit parameters (N=2)
# -----------------------------
# Assumption: we work in the rotating frame.
# H0 then represents the frequency detuning.

N = 2              # Dimension of the system (qubit: |0>, |1>)
detuning = 0.0     # 0.0 = perfect resonance (drive frequency = qubit frequency)

# Pauli operators through annihilation (b is equivalent to sigma_minus for N=2)
b = qt.destroy(N)

# Static Hamiltonian (in the rotating frame)
# H0 = detuning * b†b
H0 = detuning * b.dag() * b

# -----------------------------
# 2. Drive (control pulse)
# -----------------------------
# Control Hamiltonian: H_drive(t) = Omega(t)/2 * sigma_x
# The RL agent will control the parameters of this function (here Omega0 and sigma).

Omega0 = 1.0 * 2*np.pi  # Max Rabi amplitude (drive strength)
T_pulse = 8          # Pulse duration (ns or arbitrary units)
n_steps = 50
tlist = np.linspace(0, T_pulse, n_steps)

def eps_t(t, args):
    """
    Pulse envelope (Gaussian shape).
    The area under the curve sets the rotation angle (e.g. a pi pulse).
    """
    t0 = T_pulse / 2
    sigma = T_pulse / 6
    return Omega0 * np.exp(- (t - t0)**2 / (2*sigma**2))

# Coupling through the X operator (b + b†)
# H(t) = H0 + f(t) * (b + b†)
H = [H0, [b + b.dag(), eps_t]]

# -----------------------------
# 3. Noise model (decoherence)
# -----------------------------
# T1: energy relaxation time (|1> -> |0>)
# T_phi: pure dephasing time (loss of quantum coherence without energy loss)

T1 = 30.0    # Fast relaxation (to see the effect clearly)
T_phi = 20.0 # Strong pure dephasing

# Decay rates
# gamma_1 = 1.0 / T1
gamma_1 = 0
# gamma_phi = 1.0 / T_phi
gamma_phi = 0

# Jump operator for relaxation (spontaneous emission)
# L_relax = sqrt(gamma) * |0><1|
L_relax = np.sqrt(gamma_1) * b 

# Jump operator for pure dephasing
# L_phase = sqrt(gamma_phi) * sigma_z (close to 2*b†b in base 2)
# For N=2, sigma_z is diag(1, -1). Here the |1><1| projector is used for the phase noise
L_dephase = np.sqrt(gamma_phi) * b.dag() * b 

c_ops = [L_relax, L_dephase]

# -----------------------------
# 4. Initial state and observables
# -----------------------------

# Initial state |0>
psi0 = qt.basis(N, 0)
rho0 = qt.ket2dm(psi0) # Density matrix

# Observables: population of |0> and |1>
P0 = qt.basis(N, 0) * qt.basis(N, 0).dag()
P1 = qt.basis(N, 1) * qt.basis(N, 1).dag()

e_ops = [P0, P1]

# -----------------------------
# 5. Simulation (Master Equation)
# -----------------------------
# Solve the Lindblad master equation
result = qt.mesolve(H, rho0, tlist, c_ops, e_ops)

pop0 = result.expect[0] # Population |0>
pop1 = result.expect[1] # Population |1>

# -----------------------------
# 6. Analysis: fidelity metric
# -----------------------------
# Goal: an X gate (bit flip), so end with 100% in |1>
fidelity_final = pop1[-1]

print(f"--- CALIBRATION RESULTS ---")
print(f"Amplitude Drive (Omega0) : {Omega0/(2*np.pi):.2f} x 2pi")
print(f"Final fidelity (Pop |1>) : {fidelity_final:.4f}")
print(f"Error (1 - F)            : {1 - fidelity_final:.4f}")

# -----------------------------
# 7. Visualisation
# -----------------------------


plt.figure(figsize=(8, 5))
plt.plot(tlist, pop0, label=r"Population $|0\rangle$ (Ground)", linewidth=2)
plt.plot(tlist, pop1, label=r"Population $|1\rangle$ (Excited)", linewidth=2)
plt.xlabel("Time (ns)")
plt.ylabel("Occupation probability")
plt.title(f"Noisy qubit dynamics (T1={T1}, Tphi={T_phi})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
