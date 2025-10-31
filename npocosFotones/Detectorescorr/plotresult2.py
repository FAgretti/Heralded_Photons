import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

# Cargar datos
df = pd.read_csv('g2_barrido_pairs.csv')

# Pivotear para obtener matriz 2D (rate vs window_duration)
pivot = df.pivot(index='rate', columns='window_duration', values='g2')
rate = pivot.index.values*1e-6
windows = pivot.columns.values*1e9
g2 = pivot.values

# Plot heatmap of g2
plt.figure(figsize=(8,6))
sns.heatmap(
    g2, 
    xticklabels=[f"{w:.0f}" for w in windows], 
    yticklabels=[f"{r:.2f}" for r in rate], 
    cmap='viridis', 
    cbar_kws={'label': 'g²'},
    vmax=1
)
plt.xlabel('Ventana de integración (ns)', fontsize=18)
plt.ylabel('Frecuencia (MHz)', fontsize=18)
plt.title('g² vs rate y ventana de integración', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()


#plot a slice of the heatmap
slice_index = 8  # Cambia este valor para seleccionar diferentes cortes
plt.figure(figsize=(8,4))
plt.plot(windows, g2[slice_index, :], marker='o')
plt.xlabel('Ventana de integración (ns)', fontsize=18)
plt.ylabel('g²', fontsize=18)
plt.title(f'g² para un rate = {rate[slice_index]:.2f} MHz', fontsize=18)
plt.xscale('log')
plt.xticks(windows, [f"{w:.0f}" for w in windows], fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.tight_layout()
plt.show()