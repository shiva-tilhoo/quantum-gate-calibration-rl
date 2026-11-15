import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# -----------------------------
# 1. Paramètres du Qubit (N=2)
# -----------------------------
# Hypothèse : On travaille dans le référentiel tournant (Rotating Frame).
# H0 représente alors le désaccord en fréquence (detuning).

N = 2              # Dimension du système (Qubit : |0>, |1>)
detuning = 0.0     # 0.0 = Résonance parfaite (la fréquence du drive = fréquence du qubit)

# Opérateurs Pauli via annihilation (b équivalent à sigma_minus pour N=2)
b = qt.destroy(N)

# Hamiltonien statique (dans le référentiel tournant)
# H0 = detuning * b†b
H0 = detuning * b.dag() * b

# -----------------------------
# 2. Drive (Impulsion de contrôle)
# -----------------------------
# Hamiltonien de contrôle : H_drive(t) = Omega(t)/2 * sigma_x
# L'agent IA contrôlera les paramètres de cette fonction (ici Omega0 et sigma).

Omega0 = 1.0 * 2*np.pi  # Amplitude de Rabi max (Force du drive)
T_pulse = 8          # Durée de l'impulsion (ns ou unités arb.)
n_steps = 50
tlist = np.linspace(0, T_pulse, n_steps)

def eps_t(t, args):
    """
    Enveloppe de l'impulsion (Forme Gaussienne).
    L'aire sous la courbe détermine l'angle de rotation (ex: Pi-pulse).
    """
    t0 = T_pulse / 2
    sigma = T_pulse / 6
    return Omega0 * np.exp(- (t - t0)**2 / (2*sigma**2))

# Couplage via l'opérateur X (b + b†)
# H(t) = H0 + f(t) * (b + b†)
H = [H0, [b + b.dag(), eps_t]]

# -----------------------------
# 3. Modélisation du Bruit (Décohérence)
# -----------------------------
# T1 : Temps de relaxation énergétique (|1> -> |0>)
# T_phi : Temps de déphasage pur (perte de cohérence quantique sans perte d'énergie)

T1 = 30.0    # Relaxation rapide (pour bien voir l'effet)
T_phi = 20.0 # Déphasage pur sévère

# Taux de décroissance
# gamma_1 = 1.0 / T1
gamma_1 = 0
# gamma_phi = 1.0 / T_phi
gamma_phi = 0

# Opérateur de saut pour la relaxation (Spontaneous emission)
# L_relax = sqrt(gamma) * |0><1|
L_relax = np.sqrt(gamma_1) * b 

# Opérateur de saut pour le déphasage pur
# L_phase = sqrt(gamma_phi) * sigma_z (proche de 2*b†b en base 2)
# Pour N=2, sigma_z est diag(1, -1). Ici on utilise le projecteur |1><1| pour le bruit de phase
L_dephase = np.sqrt(gamma_phi) * b.dag() * b 

c_ops = [L_relax, L_dephase]

# -----------------------------
# 4. État initial et Observables
# -----------------------------

# État initial |0>
psi0 = qt.basis(N, 0)
rho0 = qt.ket2dm(psi0) # Matrice densité

# Observables : Population dans |0> et |1>
P0 = qt.basis(N, 0) * qt.basis(N, 0).dag()
P1 = qt.basis(N, 1) * qt.basis(N, 1).dag()

e_ops = [P0, P1]

# -----------------------------
# 5. Simulation (Master Equation)
# -----------------------------
# Résolution de l'équation maîtresse de Lindblad
result = qt.mesolve(H, rho0, tlist, c_ops, e_ops)

pop0 = result.expect[0] # Population |0>
pop1 = result.expect[1] # Population |1>

# -----------------------------
# 6. Analyse : Métrique de Fidélité
# -----------------------------
# Objectif : Faire une porte X (Bit-flip), donc finir à 100% dans |1>
fidelity_final = pop1[-1]

print(f"--- RÉSULTATS DE LA CALIBRATION ---")
print(f"Amplitude Drive (Omega0) : {Omega0/(2*np.pi):.2f} x 2pi")
print(f"Fidélité finale (Pop |1>) : {fidelity_final:.4f}")
print(f"Erreur (1 - F)           : {1 - fidelity_final:.4f}")

# -----------------------------
# 7. Visualisation
# -----------------------------


plt.figure(figsize=(8, 5))
plt.plot(tlist, pop0, label=r"Population $|0\rangle$ (Ground)", linewidth=2)
plt.plot(tlist, pop1, label=r"Population $|1\rangle$ (Excited)", linewidth=2)
plt.xlabel("Temps (ns)")
plt.ylabel("Probabilité d'occupation")
plt.title(f"Dynamique Qubit Bruité (T1={T1}, Tphi={T_phi})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
