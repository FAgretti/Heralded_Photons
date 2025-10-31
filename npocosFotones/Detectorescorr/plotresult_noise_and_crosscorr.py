import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- HERALDED WITH NOISE ---

# El archivo debe tener columnas: rate, window_duration, p_dark, p_click_heralded

heralded_file = 'heralded_with_noise.csv'
df_heralded = pd.read_csv(heralded_file)
print('Contenido de heralded_with_noise.csv:')
print(df_heralded.head(20).to_string(index=False))  # Solo las primeras 20 filas para no saturar

# Agrupar y promediar si hay duplicados
df_heralded_grouped = df_heralded.groupby(['rate', 'window_duration'], as_index=False).mean()

pivot_heralded = df_heralded_grouped.pivot(index='rate', columns='window_duration', values='p_click_heralded')
rates = pivot_heralded.index.values
windows = pivot_heralded.columns.values
pdark = df_heralded.pivot(index='rate', columns='window_duration', values='p_dark').values
g2 = pivot_heralded.values

plt.figure(figsize=(8,6))
sns.heatmap(
    g2,
    xticklabels=["{:.0e}".format(w) for w in windows],
    yticklabels=["{:.0e}".format(r) for r in rates],
    cmap='viridis',
    cbar_kws={'label': 'g² heralded (con ruido)'},
    vmax=1
)
plt.xlabel('Ventana de integración (s)', fontsize=16)
plt.ylabel('Tasa de pares (Hz)', fontsize=16)
plt.title('g²(0) heralded vs tasa de pares y ventana (con ruido)', fontsize=16)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(
    pdark,
    xticklabels=["{:.0e}".format(w) for w in windows],
    yticklabels=["{:.0e}".format(r) for r in rates],
    cmap='viridis',
    cbar_kws={'label': 'p_dark (con ruido)'},
    vmax=1
)
plt.xlabel('Ventana de integración (s)', fontsize=16)
plt.ylabel('Tasa de pares (Hz)', fontsize=16)
plt.title('p_dark vs tasa de pares y ventana (con ruido)', fontsize=16)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Cortes para diferentes tasas
for idx, rate in enumerate(rates[::max(1, len(rates)//4)]):
    plt.plot(windows, g2[idx, :], marker='o', label=f'rate={rate:.0e}')
plt.xscale('log')
plt.xlabel('Ventana de integración (s)', fontsize=16)
plt.ylabel('g² heralded (con ruido)', fontsize=16)
plt.title('Cortes de g²(0) para distintas tasas', fontsize=16)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# --- CROSS CORRELATION ---
# El archivo debe tener columnas: rate, window_duration, g2_cross
crosscorr_file = 'cross_correlation.csv'
df_cross = pd.read_csv(crosscorr_file)

pivot_cross = df_cross.pivot(index='rate', columns='window_duration', values='g2_cross')
rates_c = pivot_cross.index.values
windows_c = pivot_cross.columns.values
cc = pivot_cross.values

plt.figure(figsize=(8,6))
sns.heatmap(
    cc,
    xticklabels=["{:.0e}".format(w) for w in windows_c],
    yticklabels=["{:.0e}".format(r) for r in rates_c],
    cmap='magma',
    cbar_kws={'label': 'Correlación cruzada'},
)
plt.xlabel('Ventana de integración (s)', fontsize=16)
plt.ylabel('Tasa de pares (Hz)', fontsize=16)
plt.title('Correlación cruzada vs tasa de pares y ventana', fontsize=16)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Cortes para diferentes tasas
for idx, rate in enumerate(rates_c[::max(1, len(rates_c)//4)]):
    plt.plot(windows_c, cc[idx, :], marker='o', label=f'rate={rate:.0e}')
plt.xscale('log')
plt.xlabel('Ventana de integración (s)', fontsize=16)
plt.ylabel('Correlación cruzada', fontsize=16)
plt.title('Cortes de correlación cruzada para distintas tasas', fontsize=16)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
