import numpy as np
import matplotlib.pyplot as plt

# Вихідні дані

N = 5            # кількість щілин
d = 4.0          # відстань між щілинами, см
a = 2.0          # довжина щілини, см
lambda0 = 2.8    # довжина хвилі у вільному просторі, см
lambda_wg = 3.53 # довжина хвилі у хвилеводі, см

# Кут θ від -90° до +90°

theta_deg = np.linspace(-90.0, 90.0, 4001)
theta = np.radians(theta_deg)

# Одиночна щілина

cos_theta = np.cos(theta)
sin_theta = np.sin(theta)
cos_theta_safe = np.where(np.abs(cos_theta) < 1e-12, 1e-12, cos_theta)
F1H = np.cos((np.pi / 2.0) * sin_theta) / cos_theta_safe

# Множник решітки FᴴC(θ)

k = 2 * np.pi / lambda0
phi = k * d * sin_theta - 2.0 * np.pi * d / lambda_wg
eps = 1e-12
den = np.sin(phi / 2.0)
safe = np.abs(den) > eps
FHC = np.empty_like(phi)
FHC[safe] = np.sin((N/2.0) * phi[safe]) / (N * den[safe])
FHC[~safe] = 1.0

# Повна ДС |F_H(θ)|

FH = np.abs(F1H * FHC)
FH_norm = FH / np.max(FH)

F1H_norm = np.abs(F1H) / np.max(np.abs(F1H))
FHC_norm = np.abs(FHC) / np.max(np.abs(FHC))


# Локальні мінімуми та максимуми

local_min_idxs = np.where((FH_norm[1:-1] < FH_norm[:-2]) & (FH_norm[1:-1] < FH_norm[2:]))[0] + 1
local_max_idxs = np.where((FH_norm[1:-1] > FH_norm[:-2]) & (FH_norm[1:-1] > FH_norm[2:]))[0] + 1

tol_zero = 1e-3
null_idxs = local_min_idxs[FH_norm[local_min_idxs] <= tol_zero]
null_thetas = theta_deg[null_idxs]

imax = np.argmax(FH_norm)
local_max_idxs = local_max_idxs[local_max_idxs != imax]

# Побудова графіка

plt.figure(figsize=(12, 7))

plt.plot(theta_deg, FH_norm, color='blue', linewidth=2, label='|F_H(θ)| (норм.)')

plt.plot(theta_deg, F1H_norm, '--', color='orange', linewidth=1.5, label='|F₁ᴴ(θ)| (норм.)')
plt.plot(theta_deg, FHC_norm, ':', color='green', linewidth=1.5, label='|FᴴC(θ)| (норм.)')

for t in null_thetas:
    plt.axvline(t, color='gray', linestyle=':', alpha=0.6)

plt.scatter(theta_deg[local_min_idxs], FH_norm[local_min_idxs],
            color='red', s=30, zorder=5, label='Локальні мінімуми')

plt.scatter(theta_deg[local_max_idxs], FH_norm[local_max_idxs],
            color='yellow', edgecolor='black', s=40, zorder=5, label='Бічні максимуми')

plt.xlabel('θ, градуси', fontsize=12)
plt.ylabel('Нормована ДС |F(θ)|', fontsize=12)
plt.title('Діаграма спрямованості у площині H (варіант 17)\n'
          'F₁ᴴ(θ) = cos(π/2 ⋅ sin θ) / cos θ', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=10)
plt.xlim(-90, 90)
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()


print("Напрямки нульових випромінювань (θ у градусах):", np.round(null_thetas, 2))
print("Локальні мінімуми (θ):", np.round(theta_deg[local_min_idxs], 2))
print("Бічні максимуми (θ):", np.round(theta_deg[local_max_idxs], 2))