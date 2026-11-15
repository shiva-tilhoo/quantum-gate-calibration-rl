import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# -----------------------------
# 1. Paramètres du qutrit
# -----------------------------

N = 3                  # dimension du système (qutrit)
w = 5.0 * 2*np.pi      # fréquence "centrale" (unités arbitraires)
alpha = -0.3 * 2*np.pi # anharmonicité (niveaux non équidistants, transmon-like)

# Opérateur d'annihilation (b) et création (b†)
b = qt.destroy(N)

# Hamiltonien libre (oscillateur anharmonique type Duffing)
H0 = w * b.dag()*b + 0.5 * alpha * b.dag()*b.dag()*b*b

# -----------------------------
# 2. Drive (impulsion de contrôle)
# -----------------------------
#
# On applique un drive sur (b + b†) avec amplitude dépendant du temps : eps(t)
# L'idée : simuler une impulsion micro-onde qui vient exciter le qubit.

Omega0 = 0.1 * 2*np.pi  # amplitude max du drive
T_pulse = 20.0          # durée totale de la simulation
n_steps = 1000
tlist = np.linspace(0, T_pulse, n_steps)

def eps_t(t, args):
    """
    Impulsion gaussienne centrée.
    Tu peux changer la forme (sinus, carré, etc).
    """
    t0 = T_pulse / 2
    sigma = T_pulse / 6
    return Omega0 * np.exp(- (t - t0)**2 / (2*sigma**2))

# Hamiltonien de drive : H_drive(t) = eps(t) * (b + b†)
H_drive = [b + b.dag(), eps_t]

# Hamiltonien total dépendant du temps
H = [H0, H_drive]

# -----------------------------
# 3. Décohérence (bruit T1 / T2)
# -----------------------------
#
# T1 : relaxation (|1> -> |0>, |2> -> |1>)
# T2 : déphasage (perte de cohérence) sur |1> et |2>

# T1 et T2 en "temps" (unités arbitraires)
T1_10 = 50.0   # relaxation 1 -> 0
T1_21 = 40.0   # relaxation 2 -> 1
T2_1  = 1.0  # déphasage niveau |1>
T2_2  = 25.0   # déphasage niveau |2>

gamma_10 = 1.0 / T1_10
gamma_21 = 1.0 / T1_21
gamma_phi1 = 1.0 / T2_1
gamma_phi2 = 1.0 / T2_2

# Opérateurs de saut de relaxation
e0 = qt.basis(N, 0)
e1 = qt.basis(N, 1)
e2 = qt.basis(N, 2)

L_10 = np.sqrt(gamma_10) * e0 * e1.dag()  # |1> -> |0>
L_21 = np.sqrt(gamma_21) * e1 * e2.dag()  # |2> -> |1>

# Opérateurs de déphasage (on enlève la trace pour un pur déphasage)
P1 = e1 * e1.dag()
P2 = e2 * e2.dag()
L_phi1 = np.sqrt(gamma_phi1) * P1
L_phi2 = np.sqrt(gamma_phi2) * P2

c_ops = [L_10, L_21, L_phi1, L_phi2]

# -----------------------------
# 4. État initial et observables
# -----------------------------

# On part de |0>
psi0 = e0
rho0 = psi0 * psi0.dag()

# Observables : projecteurs sur |0>, |1>, |2>
P0 = e0 * e0.dag()
P1 = e1 * e1.dag()
P2 = e2 * e2.dag()

e_ops = [P0, P1, P2]

# -----------------------------
# 5. Simulation de la dynamique
# -----------------------------

result = qt.mesolve(H, rho0, tlist, c_ops, e_ops)

pop0 = np.array(result.expect[0])  # population sur |0>
pop1 = np.array(result.expect[1])  # population sur |1>
pop2 = np.array(result.expect[2])  # population sur |2> (fuite)

# -----------------------------
# 6. Analyse simple : fuite & "fidélité"
# -----------------------------
#
# Si on voulait par exemple réaliser une porte X complète |0> -> |1>,
# on peut approximer la fidélité à la fin comme : F = population sur |1>
# (en négligeant la phase pour cette première étude simple).

F_final = pop1[-1]
leak_final = pop2[-1]

print(f"Fidélité approximative (|0> -> |1>) à t = {T_pulse:.2f} : F = {F_final:.4f}")
print(f"Fuite vers le niveau |2> à t = {T_pulse:.2f} : L = {leak_final:.4f}")

# -----------------------------
# 7. Tracé des populations
# -----------------------------

plt.figure()
plt.plot(tlist, pop0, label="P(|0>)")
plt.plot(tlist, pop1, label="P(|1>)")
plt.plot(tlist, pop2, label="P(|2>) (fuite)")
plt.xlabel("Temps (unités arbitraires)")
plt.ylabel("Population")
plt.legend()
plt.title("Dynamique d'un qutrit avec drive + décohérence")
plt.grid(True)
plt.show()
