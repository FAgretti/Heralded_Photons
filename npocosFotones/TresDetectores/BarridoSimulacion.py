# BarridoSimulacion.py

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import Simulacion as sim
import multiprocessing as mp

def simular_una_eficiencia(args):
    eff, dark_counts, tasa_pares, duracion_total, ventana = args
    resultados = []
    for dc in dark_counts:
        g2_valor = sim.medir_g2(tasa_pares, float(eff), float(dc), duracion_total, ventana)
        resultados.append(g2_valor)
    return resultados

def barrido_eficiencia_darkcounts_paralelo(
    tasa_pares=1e5,
    duracion_total=0.01,
    ventana=1e-9,
    eficiencias=np.linspace(0.6, 1.0, 20),
    dark_counts=np.linspace(0, 400, 20)
):
    """
    Barrido sobre diferentes eficiencias y tasas de cuentas oscuras utilizando procesamiento paralelo.
    """
    args = [(eff, dark_counts, tasa_pares, duracion_total, ventana) for eff in eficiencias]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        resultados = pool.map(simular_una_eficiencia, args)

    g2_mapa = np.array(resultados)
    return eficiencias, dark_counts, g2_mapa

def graficar_resultados(eficiencias, dark_counts, g2_mapa, filename='resultados.png'):
    plt.figure(figsize=(8,6))
    plt.imshow(g2_mapa, extent=[dark_counts.min(), dark_counts.max(), eficiencias.min(), eficiencias.max()],
               origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label=r'$g^{(2)}(0)$')
    plt.xlabel('Cuentas oscuras (cps)')
    plt.ylabel('Eficiencia')
    plt.title(r'Mapa de $g^{(2)}(0)$ para fuente de pares')
    plt.savefig(filename, dpi=300)
    plt.show()

def guardar_datos(eficiencias, dark_counts, g2_mapa, filename='datos_simulacion.npz'):
    np.savez(filename, eficiencias=eficiencias, dark_counts=dark_counts, g2=g2_mapa)

if __name__ == "__main__":
    eficiencias, dark_counts, g2_mapa = barrido_eficiencia_darkcounts_paralelo()

    guardar_datos(eficiencias, dark_counts, g2_mapa)
    graficar_resultados(eficiencias, dark_counts, g2_mapa)
