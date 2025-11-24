import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ПАРАМЕТРИ
l, lam, d, xi, h = 23.7, 2.8, 2.3, 1.06, 16.0
l_lam = l / lam
pi_l_lam = np.pi * l_lam
K = (xi - 1) / np.sin(pi_l_lam * (xi - 1))

theta_deg = np.arange(0, 91, 1)
theta_rad = np.deg2rad(theta_deg)
cos_theta = np.cos(theta_rad)
sin_theta = np.sin(theta_rad)

# F_b(θ)
delta = xi - cos_theta
arg = pi_l_lam * delta
Fb = K * np.sin(arg) / delta
Fb_abs = np.abs(Fb)

# ТАБЛИЦІ F1H ТА F1E
theta_table = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
F1H_table = np.array([1.000, 0.975, 0.906, 0.806, 0.693, 0.569, 0.457, 0.389, 0.349, 0.321])
F1E_table = np.array([1.000, 0.961, 0.859, 0.771, 0.653, 0.365, 0.237, 0.131, 0.061, 0.000])

F1H = np.interp(theta_deg, theta_table, F1H_table)
F1E = np.interp(theta_deg, theta_table, F1E_table)


def plot_and_save(F_total_norm, F1, Fa_abs, title, filename, plane, formula_note=""):
    fig, ax = plt.subplots(figsize=(13, 7))

    # Побудова графіків
    ax.plot(theta_deg, Fb_abs, '#D2691E', linewidth=2.5, label='|F_b(θ)|')
    if F1 is not None:
        ax.plot(theta_deg, F1, '#228B22', linewidth=2.5, label=f'F₁{plane}(θ)')
    if Fa_abs is not None:
        ax.plot(theta_deg, Fa_abs, '#8B4513', linewidth=2.5, label='|cos(πh/λ · sinθ)|')
    ax.plot(theta_deg, F_total_norm, '#1E90FF', linewidth=3, label=f'F(θ) — {plane}')
    ax.axhline(0.707, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    # Рівень 0.707
    theta_707 = None

    idx_707_drop = np.where(F_total_norm < 0.707)[0]

    if len(idx_707_drop) > 0:
        idx_low = idx_707_drop[0]
        idx_high = idx_low - 1

        if idx_high >= 0 and idx_low < len(theta_deg):
            theta_high = theta_deg[idx_high]
            theta_low = theta_deg[idx_low]
            F_high = F_total_norm[idx_high]
            F_low = F_total_norm[idx_low]

            # ЛІНІЙНА ІНТЕРПОЛЯЦІЯ
            theta_707_precise = theta_high + (0.707 - F_high) * (theta_low - theta_high) / (F_low - F_high)
            theta_707 = round(theta_707_precise, 1)

            # ВІДОБРАЖЕННЯ
            ax.axvline(theta_707_precise, color='green', linestyle=':', linewidth=2)
            ax.text(theta_707_precise + 1, 0.72, '0.707', color='red', fontsize=10)
            ax.text(theta_707_precise + 1, 0.65, f'{theta_707}°', color='green', fontsize=10)
        else:
            theta_707 = 0.0
    else:
        theta_707 = 90.0

    # Пошук піків
    minima_idx, _ = find_peaks(-F_total_norm, prominence=0.0001)
    zeros_theta = theta_deg[minima_idx]

    # нулі
    zeros_theta = zeros_theta[zeros_theta > 0]

    ax.scatter(zeros_theta, F_total_norm[minima_idx][F_total_norm[minima_idx] < 0.1],
               color='black', s=50, zorder=5)

    for z in zeros_theta:
        ax.text(z, 0.05, f'{z}°', color='black', ha='center', fontsize=9, weight='bold')

    # Максимуми
    # Пошук піків
    peaks_idx, _ = find_peaks(F_total_norm, prominence=0.0001)
    peaks_theta = theta_deg[peaks_idx]
    peaks_level = F_total_norm[peaks_idx]

    # головний максимум для візуалізації
    side_lobes_idx = peaks_theta > 0
    side_lobes_theta = peaks_theta[side_lobes_idx]
    side_lobes_level = peaks_level[side_lobes_idx]

    ax.scatter(side_lobes_theta, side_lobes_level, color='gold', s=60, zorder=5, edgecolors='black')

    # Відображення кутів для бічних пелюсток
    for i, p in enumerate(side_lobes_theta):
        ax.text(p, side_lobes_level[i] + 0.05, f'{p}°', color='gold', ha='center', fontsize=9, weight='bold')

    rbp_theta = 30
    rbp_level = F_total_norm[30]
    rbp_db = 20 * np.log10(rbp_level) if rbp_level > 0 else -np.inf

    if theta_707 is not None:
        summary = (f"Максимум ДС при: θ = 0°\n"
                   f"Рівень 0.707 при: **θ = {theta_707}°** (точність до 0.1°)\n"
                   f"ШГП: {theta_707}° (половина)\n"
                   f"**Повна ширина: {2 * theta_707}°**\n"
                   f"РБП при: θ ≈ {rbp_theta}°, ≈ {rbp_db:.1f} дБ\n"
                   f"Напрямки нулів: {', '.join([f'{z:.0f}°' for z in zeros_theta]) or 'немає'}\n"
                   f"Макс. бок. пелюсток: {', '.join([f'{p:.0f}°' for p in side_lobes_theta]) or 'немає'}")
    else:
        summary = "Помилка розрахунку ШГП."

    ax.text(45, 0.35, summary,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
            fontsize=8.5, ha='center', va='center', linespacing=1.3)

    ax.set_title(f'{title}\n'
                 f'λ = {lam} см, l = {l} см, d = {d} см, ξ = {xi}'
                 f'{", h = 16 см" if "двох" in title else ""}\n'
                 f'{formula_note}', fontsize=13, pad=15)
    ax.set_xlabel('θ, °')
    ax.set_ylabel('F(θ), |F(θ)|')
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"ЗБЕРЕЖЕНО: {filename}")



# 1. ОДНОСТРИЖНЕВА — ПЛОЩИНА H
F_H1 = Fb_abs * F1H
F_H1_norm = F_H1 / np.max(F_H1)
plot_and_save(F_H1_norm, F1H, None,
              "ДС однострижневої антени — площина H",
              "ДС_однострижнева_площина_H_v2.png", "H",
              "F_H(θ) = |F_b(θ)| · F₁H(θ)")

# 2. ОДНОСТРИЖНЕВА — ПЛОЩИНА E
F_E1 = Fb_abs * F1E
F_E1_norm = F_E1 / np.max(F_E1)
plot_and_save(F_E1_norm, F1E, None,
              "ДС однострижневої антени — площина E",
              "ДС_однострижнева_площина_E_v2.png", "E",
              "F_E(θ) = |F_b(θ)| · F₁E(θ)")

# 3. ДВОХСТРИЖНЕВА — ПЛОЩИНА H
Fa_H = np.cos((np.pi * h / lam) * sin_theta)
Fa_H_abs = np.abs(Fa_H)
F_H2 = Fb_abs * Fa_H_abs
F_H2_norm = F_H2 / np.max(F_H2)
plot_and_save(F_H2_norm, None, Fa_H_abs,
              "ДС двохстрижневої антени — площина H",
              "ДС_двохстрижнева_площина_H_v2.png", "H",
              "F_H(θ) = |F_b(θ)| · |cos(πh/λ · sinθ)|")

# 4. ДВОХСТРИЖНЕВА — ПЛОЩИНА E
Fa_E = np.cos((np.pi * h / lam) * sin_theta)
Fa_E_abs = np.abs(Fa_E)
F_E2 = Fb_abs * Fa_E_abs * cos_theta
F_E2_norm = F_E2 / np.max(F_E2)
plot_and_save(F_E2_norm, None, Fa_E_abs,
              "ДС двохстрижневої антени — площина E",
              "ДС_двохстрижнева_площина_E_v2.png", "E",
              "F_E(θ) = |F_b(θ)| · |cos(πh/λ · sinθ)| · cosθ  ← (4.29)")

# =====================================================
print("\nУСІ 4 ГРАФІКИ ЗБЕРЕЖЕНО!")