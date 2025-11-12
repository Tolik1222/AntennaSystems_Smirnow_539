import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.signal import find_peaks

# --- Параметри варіанту 17 ---
lam = 0.028
D = 0.7
f = 0.3
k = 2*np.pi/lam
R0 = D/2
p = 2*f
v = 3.5*R0/p
v1_5 = 1.5*v

# --- Кутова сітка ---
theta_deg = np.linspace(0, 90, 50001)  # дуже дрібний крок для вузьких пелюсток
theta = np.deg2rad(theta_deg)
u = k * R0 * np.sin(theta)

# --- Функції Бесселя ---
J0_u = jv(0, u)
J1_u = jv(1, u)
J2_u = jv(2, u)
J0_v = jv(0, v)
J1_v = jv(1, v)
J1_1_5v = jv(1, v1_5)
J2_1_5v = jv(2, v1_5)

# --- Безпечне ділення ---
def safe_div(n, d):
    out = np.zeros_like(n)
    mask = np.abs(d) > 1e-12
    out[mask] = n[mask] / d[mask]
    return out

# --- Обчислення термів ---
term_a = safe_div(v*J1_v*J0_u - u*J1_u*J0_v, v**2 - u**2)
term_b = np.zeros_like(u)
mask = ~np.isclose(u, 0.0)
term_b[mask] = J1_u[mask]/u[mask]
term_b[~mask] = 0.5
term_c = safe_div(u*J1_u*J2_1_5v - v1_5*J1_1_5v*J2_u, v1_5**2 - u**2)

# --- Чисельники ---
numer_H = 0.74*term_a + 0.26*term_b - 0.25*term_c
numer_E = 0.74*term_a + 0.26*term_b + 0.25*term_c
denom = 0.74*(J1_v/v) + 0.13

# --- Поля (без модуля) ---
F_H = (np.cos(theta/2)**2)*(numer_H/denom)
F_E = (np.cos(theta/2)**2)*(numer_E/denom)

# --- Нормування для HPBW та пошуку максимумів ---
FHn = np.abs(F_H)/np.max(np.abs(F_H))
FEn = np.abs(F_E)/np.max(np.abs(F_E))

# --- Пошук локальних мінімумів і бічних максимумів ---
mins_H, _ = find_peaks(-FHn)
mins_E, _ = find_peaks(-FEn)
side_peaks_H, _ = find_peaks(FHn)
side_peaks_E, _ = find_peaks(FEn)

# --- Функція для HPBW ---
def find_HPBW(pattern, angles_deg):
    hp_level = 0.707
    max_idx = np.argmax(pattern)
    left = right = None
    # шукати зліва
    for i in range(max_idx, 0, -1):
        if pattern[i] >= hp_level and pattern[i-1] < hp_level:
            left = np.interp(hp_level, [pattern[i-1], pattern[i]], [angles_deg[i-1], angles_deg[i]])
            break
    # шукати справа
    for i in range(max_idx, len(pattern)-1):
        if pattern[i] >= hp_level and pattern[i+1] < hp_level:
            right = np.interp(hp_level, [pattern[i], pattern[i+1]], [angles_deg[i], angles_deg[i+1]])
            break
    # fallback для вузьких пелюсток
    if left is None or right is None:
        below = np.where(pattern < hp_level)[0]
        if below.size > 0:
            left = angles_deg[0]
            right = angles_deg[below[0]]
            return (right - left) * 2, left, right
        else:
            return None, None, None
    return (right - left) * 2, left, right

HPBW_H, left_H_deg, right_H_deg = find_HPBW(FHn, theta_deg)
HPBW_E, left_E_deg, right_E_deg = find_HPBW(FEn, theta_deg)

# --- Значення по осі X ---
left_H_u = k*R0*np.sin(np.deg2rad(left_H_deg)) if left_H_deg is not None else None
right_H_u = k*R0*np.sin(np.deg2rad(right_H_deg)) if right_H_deg is not None else None
left_E_u = k*R0*np.sin(np.deg2rad(left_E_deg)) if left_E_deg is not None else None
right_E_u = k*R0*np.sin(np.deg2rad(right_E_deg)) if right_E_deg is not None else None

## --- Залишаємо лише парні бічні максимуми ---
side_peaks_H_sel = [side_peaks_H[i] for i in range(len(side_peaks_H)) if i % 2 == 1]
side_peaks_E_sel = [side_peaks_E[i] for i in range(len(side_peaks_E)) if i % 2 == 1]

# --- Побудова графіка ---
plt.figure(figsize=(12,6))
plt.plot(u, F_H, label='F_H(θ)', color='blue')
plt.plot(u, F_E, label='F_E(θ)', color='orange', linestyle='--')

# Горизонтальна лінія рівня 0.707
plt.axhline(0.707, color='black', linestyle=':', label='Рівень 0.707')

# Вертикальні лінії HPBW
if left_H_u is not None and right_H_u is not None:
    plt.axvline(right_H_u, color='blue', linestyle=':')

if left_E_u is not None and right_E_u is not None:

    plt.axvline(right_E_u, color='orange', linestyle=':')

# Локальні мінімумі
plt.scatter(u[mins_H], F_H[mins_H], color='red', s=25, label='Локальні мінімумі H')
plt.scatter(u[mins_E], F_E[mins_E], color='red', s=25, marker='x', label='Локальні мінімумі E')

## Бічні максимумі (тільки парні)
plt.scatter(u[side_peaks_H_sel], F_H[side_peaks_H_sel], color='yellow', s=50, label='Бічні максимумі H (парні)')
plt.scatter(u[side_peaks_E_sel], F_E[side_peaks_E_sel], color='green', s=50, label='Бічні максимумі E (парні)')

plt.xlabel('k * R0 * sin(θ)')
plt.ylabel('Амплітуда поля (без модуля)')
plt.title('Діаграма спрямованості (0°–90°) — H і E площини')
plt.grid(True)
plt.legend()
plt.show()

# --- Вивід результатів ---
print("=== Результати розрахунків ===")
print(f"ШГП H-площини (подвоєна): {HPBW_H:.4f}°" if HPBW_H else "HPBW для H-площини не знайдено")
print(f"ШГП E-площини (подвоєна): {HPBW_E:.4f}°" if HPBW_E else "HPBW для E-площини не знайдено")

print("\nБічні максимумі H-площини:")
for p in side_peaks_H:
    print(f"θ = {theta_deg[p]:.4f}°, kR0sinθ = {u[p]:.4f}, F_H = {F_H[p]:.4f}")

print("\nБічні максимумі E-площини:")
for p in side_peaks_E:
    print(f"θ = {theta_deg[p]:.4f}°, kR0sinθ = {u[p]:.4f}, F_E = {F_E[p]:.4f}")
