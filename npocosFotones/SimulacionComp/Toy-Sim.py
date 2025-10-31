import cupy as cp
import matplotlib.pyplot as plt
from tqdm import trange
import Toy as Toy
import numpy as np
from tqdm import tqdm



# --- Sweep parameters ---
eficiencia = np.linspace(0.5, 0.1, 5)  # More points for smoother curves
dark_count = np.linspace(0, 1000, 50)   # More points for smoother curves
n_bins = 10000
bin_duration = 1.0
ventana = 0.1
duracion = n_bins * bin_duration
n_avg = 5  # More averages for smoother results

g2 = np.zeros((len(eficiencia), len(dark_count)))

for i, eff in enumerate(tqdm(eficiencia, desc="Barrido eficiencia")):
    for j, dark in enumerate(dark_count):
        g2_sum = 0
        avg_count = n_avg if eff > 0.1 else n_avg * 4
        for _ in range(int(avg_count)):
            g2_sum += Toy.sim(n_bins, bin_duration, ventana, dark, duracion, eff)
        g2[i, j] = g2_sum / avg_count

# --- Save results ---
np.savez('g2_sweep_results.npz', g2=g2, eficiencia=eficiencia, dark_count=dark_count, n_bins=n_bins, bin_duration=bin_duration, ventana=ventana, n_avg=n_avg)

# Plot
plt.figure(1)
for i in range(len(eficiencia)):
    plt.plot(dark_count, g2[i], label=f"Eff={eficiencia[i]:.2f}")
plt.grid()
plt.legend()
plt.xlabel('Dark counts')
plt.ylabel('g2')
plt.title('g2 vs dark counts')

plt.figure(2)
# Normalized plot
for i in range(len(eficiencia)):
    norm = eficiencia[i] * dark_count
    norm[norm == 0] = 1  # avoid division by zero
    plt.plot(dark_count, g2[i] / norm, label=f"Eff={eficiencia[i]:.2f}")
plt.grid()
plt.legend()
plt.xlabel('Dark counts')
plt.ylabel('g2 / (eff * dark_count)')
plt.title('Normalized g2')
plt.show()
