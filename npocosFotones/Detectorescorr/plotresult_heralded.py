import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
# El archivo debe tener columnas: rate, window_duration, g2_heralded
df = pd.read_csv('g2_barrido_heralded.csv')

# Pivotear para obtener matriz 2D (rate vs window_duration)
pivot = df.pivot(index='rate', columns='window_duration', values='g2_heralded')
rates = pivot.index.values
windows = pivot.columns.values
g2 = pivot.values

plt.figure(figsize=(8,6))
sns.heatmap(
    g2,
    xticklabels=["{:.0e}".format(w) for w in windows],
    yticklabels=["{:.0e}".format(r) for r in rates],
    cmap='viridis',
    cbar_kws={'label': 'g² heralded'},
    vmax=1
)
plt.xlabel('Ventana de integración (s)', fontsize=16)
plt.ylabel('Tasa de pares (Hz)', fontsize=16)
plt.title('g²(0) heralded vs tasa de pares y ventana', fontsize=16)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Graficar cortes para diferentes tasas
for idx, rate in enumerate(rates[::max(1, len(rates)//4)]):
    plt.plot(windows, g2[idx, :], marker='o', label=f'rate={rate:.0e}')
plt.xscale('log')
plt.xlabel('Ventana de integración (s)', fontsize=16)
plt.ylabel('g² heralded', fontsize=16)
plt.title('Cortes de g²(0) heralded para distintas tasas', fontsize=16)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
