import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

# Cargar datos
df = pd.read_csv('g2_barrido.csv')

# Pivotear para obtener matriz 2D (min_gap vs window_duration)
pivot = df.pivot(index='min_gap', columns='window_duration', values='g2')
min_gaps = pivot.index.values*1e9
windows = pivot.columns.values*1e9
g2 = pivot.values

# Plot heatmap of g2
plt.figure(figsize=(8,6))
sns.heatmap(
    g2, 
    xticklabels=["{:.2f}".format(w) for w in windows], 
    yticklabels=["{:.2f}".format(mg) for mg in min_gaps], 
    cmap='viridis', 
    cbar_kws={'label': 'g²'},
    vmax=1
)
plt.xlabel('Ventana de integración (ns)', fontsize=18)
plt.ylabel('Separación mínima (ns)', fontsize=18)
plt.title('g² vs separación mínima y ventana de integración', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()


#plot a slice of the heatmap
slice_index = 2  # Change this to select different slices
plt.figure(figsize=(8,4))
plt.plot(windows, g2[slice_index, :], marker='o')
plt.xlabel('Ventana de integración (ns)', fontsize=18)
plt.ylabel('g²', fontsize=18)
plt.title(f'g² para una separación mínima = {min_gaps[slice_index]:.0f} ns', fontsize=18)
# Use scientific notation and rotate x labels for readability
plt.xscale('log')
plt.xticks(windows, ["{:.0e}".format(w) for w in windows], fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.grid()
plt.tight_layout()
plt.show()