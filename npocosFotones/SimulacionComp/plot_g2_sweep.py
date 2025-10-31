import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv('g2_sweep_results.csv')

# Pivotear para obtener matriz 2D (eficiencia vs dark_count)
pivot = df.pivot(index='efficiency', columns='dark_count', values='g2')
effs = pivot.index.values
darks = pivot.columns.values
g2 = pivot.values

# --- Primer gráfico: mapa de calor (cap a 1) ---
plt.figure(figsize=(10,6))
sns.heatmap(g2, 
            xticklabels=40, 
            yticklabels=np.round(effs, 2), 
            cmap='viridis', 
            cbar_kws={'label': 'g²'},
            vmax=1)
plt.xlabel('Dark count')
plt.ylabel('Eficiencia')
plt.title('Mapa de calor de g² vs eficiencia y dark count (g² ≤ 1)')
plt.tight_layout()
plt.show()

# --- Segundo gráfico: g2 vs dark_count para eficiencias seleccionadas ---
plt.figure(figsize=(10,6))
step = max(1, len(effs)//6)  # Muestra ~6 curvas
for i, eff in enumerate(effs):
    if i % step == 0 or i == len(effs)-1:
        plt.plot(darks, g2[i, :], label=f'Eficiencia={eff:.2f}')
plt.xlabel('Cuentas oscuras', fontsize=18)
plt.ylabel('g²', fontsize=18)
plt.title('g² vs Cuentas oscuras (1000 cuentas reales)', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# # --- Opción 1: g² vs dark_count con eje secundario para fotones reales detectados ---
# plt.figure(figsize=(10,6))
# ax1 = plt.gca()
# step = max(1, len(effs)//6)
# colors = plt.cm.viridis(np.linspace(0,1,len(effs)))
# for i, eff in enumerate(effs):
#     if i % step == 0 or i == len(effs)-1:
#         ax1.plot(darks, g2[i, :], label=f'Eficiencia={eff:.2f}', color=colors[i])
# ax1.set_xlabel('Dark count')
# ax1.set_ylabel('g²')
# ax2 = ax1.twinx()
# for i, eff in enumerate(effs):
#     if i % step == 0 or i == len(effs)-1:
#         n_real = (darks) * eff  # n_bins * eficiencia (aprox)
#         ax2.plot(darks, [n_real]*len(darks), '--', color=colors[i], alpha=0.5)
# ax2.set_ylabel('Fotones reales detectados (n_bins * eficiencia)')
# plt.title('g² vs Dark count y cantidad de fotones reales detectados')
# ax1.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# # --- Opción 2: g² normalizado por eficiencia ---
# plt.figure(figsize=(10,6))
# for i, eff in enumerate(effs):
#     if i % step == 0 or i == len(effs)-1:
#         plt.plot(darks, g2[i, :] / eff, label=f'Eficiencia={eff:.2f}')
# plt.xlabel('Dark count')
# plt.ylabel('g² / eficiencia')
# plt.title('g² normalizado por eficiencia vs Dark count')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.show()