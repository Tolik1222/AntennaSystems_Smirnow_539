import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

# ===== Вхідні дані (твій варіант 17) =====
N = 8
F = 580e6
c = 299792458
lambd = c / F
d = 0.25 * lambd
k = (2 * np.pi) / lambd

print(f"λ = {lambd:.4f} м")
print(f"dср = {d:.4f} м")
print(f"k = {k:.4f} рад/м")


# ===== Функції =====
def F1E(theta):
    """ДС напівхвильового вібратора (E-площина)"""
    return abs(np.cos(np.pi/2 * np.sin(theta)) / np.cos(theta))

def FC(theta):
    """Множник решітки"""
    return abs(np.sin((N * k * d * (1 - np.cos(theta)) / 2)) /
               (N * np.sin((k * d * (1 - np.cos(theta))) / 2)))

def FE(theta):
    """ДС у площині E"""
    return F1E(theta) * FC(theta)


# ===== Розрахунок =====
theta_range = np.arange(0.01, np.pi/2, 0.0005)  # від 0 до 90°
steps_deg = np.degrees(theta_range)

F1E_vals = F1E(theta_range)
FC_vals = FC(theta_range)
FE_vals = FE(theta_range)

# Ширина головної пелюстки (на рівні 0.707)
SGP_H = 2 * steps_deg[np.argmin(abs(FC_vals - 0.707))]
SGP_E = 2 * steps_deg[np.argmin(abs(FE_vals - 0.707))]

print(f"Ширина головної пелюстки в площині H = {SGP_H:.2f}°")
print(f"Ширина головної пелюстки в площині E = {SGP_E:.2f}°")


# ===== Пошук максимумів і мінімумів =====
def find_extrema(values, steps):
    maxima, minima = [], []
    for i in range(1, len(values) - 1):
        if values[i] > values[i-1] and values[i] > values[i+1]:
            maxima.append((steps[i], values[i]))
        if values[i] < values[i-1] and values[i] < values[i+1]:
            minima.append((steps[i], values[i]))
    return maxima, minima

max_FC, min_FC = find_extrema(FC_vals, steps_deg)
max_FE, min_FE = find_extrema(FE_vals, steps_deg)

# ===== Таблиця =====
df = pd.DataFrame({
    "№": range(1, max(len(max_FC), len(max_FE)) + 1),
    "θminH (°)": [f"{x[0]:.2f}" for x in min_FC] + ["-"] * (len(max_FE) - len(min_FC)),
    "θminE (°)": [f"{x[0]:.2f}" for x in min_FE] + ["-"] * (len(max_FC) - len(min_FE)),
    "θmaxH (°)": [f"{x[0]:.2f}" for x in max_FC] + ["-"] * (len(max_FE) - len(max_FC)),
    "FH(θ)": [f"{x[1]:.3f}" for x in max_FC] + ["-"] * (len(max_FE) - len(max_FC)),
    "θmaxE (°)": [f"{x[0]:.2f}" for x in max_FE] + ["-"] * (len(max_FC) - len(max_FE)),
    "FE(θ)": [f"{x[1]:.3f}" for x in max_FE] + ["-"] * (len(max_FC) - len(max_FE))
})

print("\nТаблиця 1 – Аналіз ДС директорної антени")
print(df.to_string(index=False))


# ===== Побудова графіків =====
plt.figure(figsize=(9, 6))
plt.plot(steps_deg, F1E_vals, label="F1E(θ) – вібратор", color="gray", linestyle="--", linewidth=1.2)
plt.plot(steps_deg, FC_vals, label="FH(θ) – H площина", color="blue", linewidth=1.5)
plt.plot(steps_deg, FE_vals, label="FE(θ) – E площина", color="red", linewidth=1.5)

# Позначимо ШГП
plt.axvline(SGP_H/2, color="blue", linestyle=":", linewidth=1)
plt.axvline(SGP_E/2, color="red", linestyle=":", linewidth=1)
plt.scatter([SGP_H/2], [0.707], color="blue", marker="o")
plt.scatter([SGP_E/2], [0.707], color="red", marker="o")

plt.text(SGP_H/2+1, 0.72, f"{SGP_H:.1f}°", color="blue")
plt.text(SGP_E/2+1, 0.75, f"{SGP_E:.1f}°", color="red")

plt.title("Діаграми спрямованості директорної антени", fontsize=12, weight="bold")
plt.xlabel("Кут θ (°)", fontsize=11)
plt.ylabel("Нормовані значення F(θ)", fontsize=11)
plt.xlim(0, 90)
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
