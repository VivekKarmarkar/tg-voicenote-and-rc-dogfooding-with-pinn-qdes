"""
Toy thermal-wind model for Iowa City, May 7 2026.

Physics:
  Solar input S(t) = UV_index(t) as proxy for solar irradiance (already cloud-modulated)
  Surface temperature: C * dT_s/dt = alpha * S(t) - sigma * (T_s - T_air(t))
  Local wind: W_local(t) = k * max(T_s(t) - T_air(t), 0)^0.5

We fit C, alpha, sigma, k to match observed wind data.
Then plot predicted vs observed wind + the solar forcing.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# === Observed data: Iowa City, May 7 2026, hourly ===
hours = np.arange(24)

uv_index = np.array([
    0, 0, 0, 0, 0, 0,        # 0-5 AM
    0, 0.2, 0.7, 2.05, 3.85, 4.85,  # 6-11 AM
    6.1, 6.8, 6.95, 6.5, 5.4, 3.7,  # 12-5 PM
    2.2, 1.15, 0.3, 0, 0, 0         # 6-11 PM
])

cloud_cover = np.array([
    0, 0, 0, 0, 0, 0,        # 0-5 AM
    99, 24, 99, 81, 99, 99,   # 6-11 AM
    95, 12, 5, 99, 10, 15,    # 12-5 PM
    99, 6, 99, 100, 0, 100    # 6-11 PM
])

temp_air = np.array([
    39.3, 38.8, 36.6, 35.7, 37.3, 36.0,
    37.5, 40.9, 45.5, 50.6, 53.7, 53.6,
    54.9, 57.2, 59.6, 61.4, 64.2, 66.0,
    64.4, 63.0, 61.3, 58.2, 56.4, 54.2
])

wind_observed = np.array([
    0.9, 3.5, 3.6, 3.8, 3.5, 3.8,
    3.3, 6.9, 7.4, 12.9, 14.3, 15.5,
    15.9, 14.5, 19.1, 16.0, 14.8, 13.0,
    12.3, 9.2, 3.3, 2.2, 4.1, 4.0
])

# Solar forcing: UV index is already cloud-modulated
S = uv_index.copy()

# Interpolate to finer time steps for ODE integration
dt = 0.01  # hours
t_fine = np.arange(0, 24, dt)
S_fine = np.interp(t_fine, hours, S)
T_air_fine = np.interp(t_fine, hours, temp_air)

# Wind_global baseline (nighttime wind suggests ~3-4 mph synoptic component)
W_global = 3.5


def simulate(params):
    C, alpha, sigma, k = params
    T_s = np.zeros_like(t_fine)
    T_s[0] = T_air_fine[0]

    for i in range(1, len(t_fine)):
        dTdt = (alpha * S_fine[i-1] - sigma * (T_s[i-1] - T_air_fine[i-1])) / C
        T_s[i] = T_s[i-1] + dTdt * dt

    delta_T = np.maximum(T_s - T_air_fine, 0)
    W_local = k * np.sqrt(delta_T)
    W_total = W_global + W_local

    # Sample back to hourly
    W_hourly = np.interp(hours, t_fine, W_total)
    T_s_hourly = np.interp(hours, t_fine, T_s)
    return W_hourly, T_s_hourly, T_s, W_local, np.interp(hours, t_fine, W_local)


def cost(params):
    if any(p <= 0 for p in params):
        return 1e6
    W_hourly, _, _, _, _ = simulate(params)
    return np.sum((W_hourly - wind_observed) ** 2)


# Optimize
result = minimize(cost, x0=[2.0, 5.0, 1.0, 3.0], method='Nelder-Mead',
                  options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6})
C_opt, alpha_opt, sigma_opt, k_opt = result.x

W_pred, T_s_pred, T_s_fine, W_local_fine, W_local_hourly = simulate(result.x)

# === Plot ===
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

# Panel 1: Solar forcing
axes[0].fill_between(hours, S, alpha=0.3, color='orange', label='UV Index (cloud-modulated)')
axes[0].bar(hours, cloud_cover / 100 * max(S), alpha=0.15, color='gray',
            width=0.8, label='Cloud cover (scaled)')
axes[0].set_ylabel('UV Index')
axes[0].set_title('Solar Forcing — Iowa City, May 7 2026')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Panel 2: Surface temperature vs air temperature
axes[1].plot(hours, T_s_pred, 'r-o', markersize=4, label=f'T_surface (modeled)', linewidth=2)
axes[1].plot(hours, temp_air, 'b-s', markersize=4, label='T_air (observed)', linewidth=2)
axes[1].fill_between(hours, temp_air, T_s_pred,
                     where=T_s_pred > temp_air, alpha=0.2, color='red',
                     label='ΔT (drives wind)')
axes[1].set_ylabel('Temperature (°F)')
axes[1].set_title('Surface vs Air Temperature — Thermal Lag')
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

# Panel 3: Wind comparison
axes[2].plot(hours, wind_observed, 'k-o', markersize=5, label='Wind observed', linewidth=2)
axes[2].plot(hours, W_pred, 'r--^', markersize=5, label='Wind predicted (model)', linewidth=2)
axes[2].axhline(y=W_global, color='gray', linestyle=':', alpha=0.5, label=f'W_global = {W_global} mph')
axes[2].fill_between(hours, W_global, W_pred, alpha=0.15, color='red', label='W_local contribution')
axes[2].set_ylabel('Wind Speed (mph)')
axes[2].set_title('Predicted vs Observed Wind')
axes[2].legend(loc='upper left')
axes[2].grid(True, alpha=0.3)

# Panel 4: Decomposition
axes[3].plot(hours, wind_observed, 'k-o', markersize=5, label='Wind observed', linewidth=2)
axes[3].bar(hours, W_local_hourly, alpha=0.4, color='red', width=0.8, label='W_local (solar-driven)')
axes[3].axhline(y=W_global, color='blue', linestyle='-', alpha=0.7,
                label=f'W_global (synoptic) = {W_global} mph', linewidth=2)
axes[3].set_ylabel('Wind Speed (mph)')
axes[3].set_xlabel('Hour of Day')
axes[3].set_title('Wind Decomposition: W_total = W_global + W_local(Sun(t))')
axes[3].legend(loc='upper left')
axes[3].grid(True, alpha=0.3)
axes[3].set_xticks(range(0, 24, 2))
axes[3].set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45)

plt.tight_layout()

# Compute R²
ss_res = np.sum((wind_observed - W_pred) ** 2)
ss_tot = np.sum((wind_observed - np.mean(wind_observed)) ** 2)
r_squared = 1 - ss_res / ss_tot

# Add R² and parameters as text
fig.text(0.02, 0.01,
         f'Fitted params: C={C_opt:.2f}, α={alpha_opt:.2f}, σ={sigma_opt:.2f}, k={k_opt:.2f} | '
         f'R² = {r_squared:.3f} | W_global = {W_global} mph',
         fontsize=10, family='monospace')

output_path = '/home/vivekkarmarkar/Python Files/tg-voicenote-and-rc-dogfooding-with-pinn-qdes/weather_toy_model_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"R² = {r_squared:.4f}")
print(f"Fitted parameters: C={C_opt:.3f}, alpha={alpha_opt:.3f}, sigma={sigma_opt:.3f}, k={k_opt:.3f}")
print(f"Saved to {output_path}")
