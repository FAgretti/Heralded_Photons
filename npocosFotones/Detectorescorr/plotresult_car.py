import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos de la simulación
# El archivo debe tener columnas: rate, noise, window, car
car_file = 'car_simulation.csv'
df = pd.read_csv(car_file)

# Pivot para heatmap: CAR vs rate y window, para cada valor de noise
for noise in sorted(df['noise'].unique()):
    df_n = df[df['noise']==noise]
    pivot = df_n.pivot(index='rate', columns='window', values='car')
    rates = pivot.index.values
    windows = pivot.columns.values
    car = pivot.values
    plt.figure(figsize=(8,6))
    sns.heatmap(
        car,
        xticklabels=["{:.0e}".format(w) for w in windows],
        yticklabels=["{:.0e}".format(r) for r in rates],
        cmap='plasma',
        cbar_kws={'label': 'CAR'},
        annot=True, fmt='.1f'
    )
    plt.xlabel('Ventana de coincidencia (s)', fontsize=16)
    plt.ylabel('Tasa de pares (Hz)', fontsize=16)
    plt.title(f'CAR vs tasa de pares y ventana (ruido={int(noise)} Hz)', fontsize=16)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

# Cortes de CAR para distintas tasas de pares
for noise in sorted(df['noise'].unique()):
    df_n = df[df['noise']==noise]
    for idx, rate in enumerate(sorted(df_n['rate'].unique())[::max(1, len(df_n['rate'].unique())//4)]):
        df_r = df_n[df_n['rate']==rate]
        plt.plot(df_r['window'], df_r['car'], marker='o', label=f'rate={rate:.0e}')
    plt.xscale('log')
    plt.xlabel('Ventana de coincidencia (s)', fontsize=16)
    plt.ylabel('CAR', fontsize=16)
    plt.title(f'Cortes de CAR para distintas tasas (ruido={int(noise)} Hz)', fontsize=16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
