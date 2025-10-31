import numpy as np
import matplotlib.pyplot as plt

# --- Copia de la función de simulación principal ---
def simulate_g2(n_windows, window_duration, rate, antibunching=True, min_gap=0.5e-9, cross_talk=0.0):
    total_time = n_windows * window_duration
    # Generar tiempos de arribo
    if antibunching:
        times = []
        t = 0
        while t < total_time:
            dt = np.random.exponential(1/rate)
            if dt < min_gap:
                dt = min_gap
            t += dt
            if t < total_time:
                times.append(t)
        photon_times = np.array(times)
    else:
        n_photons = np.random.poisson(rate * total_time)
        photon_times = np.sort(np.random.uniform(0, total_time, n_photons))
    # Simular beam splitter y cross-talk
    split_probs = np.random.rand(len(photon_times))
    det1_times = photon_times[split_probs < 0.5]
    det2_times = photon_times[split_probs >= 0.5]
    both_detected = photon_times[np.random.rand(len(photon_times)) < cross_talk]
    det1_times = np.concatenate([det1_times, both_detected])
    det2_times = np.concatenate([det2_times, both_detected])
    # Bin
    bins = np.arange(0, total_time + window_duration, window_duration)
    I1, _ = np.histogram(det1_times, bins)
    I2, _ = np.histogram(det2_times, bins)
    # Calcular g2
    g2_num = np.mean(I1 * I2)
    g2_den = np.mean(I1) * np.mean(I2)
    g2 = g2_num / g2_den if g2_den > 0 else np.nan
    return g2, np.mean(I1), np.mean(I2)

# --- Barrido de parámetros ---
n_windows = 100000
rate = 1e8
cross_talk = 0.0
min_gap_list = [0.5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6]
window_list = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6]

results = np.zeros((len(min_gap_list), len(window_list)))

for i, min_gap in enumerate(min_gap_list):
    for j, window_duration in enumerate(window_list):
        g2, mean1, mean2 = simulate_g2(n_windows, window_duration, rate, antibunching=True, min_gap=min_gap, cross_talk=cross_talk)
        results[i, j] = g2
        print(f"min_gap={min_gap:.1e}, window={window_duration:.1e}, g2={g2:.3f}, mean1={mean1:.2f}, mean2={mean2:.2f}")

# --- Graficar mapa de calor ---
plt.figure(figsize=(8,6))
plt.imshow(results, origin='lower', aspect='auto',
           extent=(window_list[0], window_list[-1], min_gap_list[0], min_gap_list[-1]),
           cmap='viridis')
plt.colorbar(label='g2(0)')
plt.xlabel('Duración de ventana (s)')
plt.ylabel('min_gap antibunching (s)')
plt.title('g2(0) vs ventana y antibunching')
plt.tight_layout()
plt.show()
