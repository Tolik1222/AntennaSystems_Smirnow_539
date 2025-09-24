import numpy as np
import matplotlib.pyplot as plt

# Параметри
f = 580e6  # Частота, Гц
c = 3e8    # Швидкість світла, м/с
lambda_ = c / f  # Довжина хвилі
d = 0.25 * lambda_  # Відстань між елементами
N = 8      # Кількість елементів
theta = np.linspace(-np.pi/2, np.pi/2, 1000)  # Кути в радіанах

# Розрахунок ДС
psi = (np.pi * d / lambda_) * np.cos(theta)
F = np.abs(np.sin(N * psi) / (N * np.sin(psi)))
F_norm = F / np.max(F)  # Нормування

# Побудова графіку
plt.figure(figsize=(10, 6))
plt.plot(np.degrees(theta), F_norm, label='ДС у площині H')
plt.axhline(y=0.707, color='r', linestyle='--', label='Рівень 0.707')
plt.xlabel('Кут θ, градуси')
plt.ylabel('Нормована амплітуда')
plt.title('Діаграма спрямованості директорної антени (варіант 17)')
plt.grid(True)
plt.legend()
plt.show()

# Оцінка ширини головної пелюстки
half_power_indices = np.where(F_norm >= 0.707)[0]
theta_half_power = np.degrees(theta[half_power_indices[[0, -1]]])
beamwidth = abs(theta_half_power[1] - theta_half_power[0])
print(f"Ширина головної пелюстки: {beamwidth:.2f} градусів")

# Оцінка рівня бокових пелюсток
side_lobe_level = np.max(F_norm[F_norm < 0.707])
print(f"Рівень бокових пелюсток: {side_lobe_level:.3f}")