import cupy as cp
import matplotlib.pyplot as plt
from tqdm import trange
import Toy as Toy
import numpy as np
from tqdm import tqdm


eficiencia = np.linspace(1, 0.1, 5)
dark_count = np.linspace(0, 1000, 20)
n_bins = 3000
bin_duration = 1.0
ventana = 0.1
duracion = n_bins * bin_duration
n_avg = 15  # number of averages

g2 = np.zeros((len(eficiencia), len(dark_count)))

for i, eff in enumerate(tqdm(eficiencia, desc="Barrido eficiencia")):
    for j, dark in enumerate(dark_count):
        g2_sum = 0
        for _ in range(n_avg):
            g2_sum += Toy.sim(n_bins, bin_duration, ventana, dark, duracion, eff)
        g2[i, j] = g2_sum / n_avg

# Plot
plt.figure(1)
for i in range(len(eficiencia)):
    plt.plot(dark_count, g2[i], label=f"Eff={eficiencia[i]:.2f}")
    #plt.vlines(n_bins * eficiencia[i], 0, 1)
plt.grid()
plt.legend()

plt.figure(2)
#Quiero que en esta figura, el g2 este normalizado por el producto de las eficiencias y el numero de cuentas oscuras
for i in range(len(eficiencia)):
    plt.plot(dark_count, g2[i] / (eficiencia[i] * dark_count), label=f"Eff={eficiencia[i]:.2f}")
    #plt.vlines(n_bins * eficiencia[i], 0, 1)
plt.grid()
plt.legend()
plt.show()
