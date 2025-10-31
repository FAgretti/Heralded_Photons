# Archivo: graficos.py

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def cargar_resultados(directorio="resultados"):
    archivos = [f for f in os.listdir(directorio) if f.endswith(".pkl")]
    resultados = []
    for archivo in archivos:
        with open(os.path.join(directorio, archivo), "rb") as f:
            resultados.append(pickle.load(f))
    return resultados

def promediar_resultados(lista_resultados):
    claves = lista_resultados[0].keys()
    promedio = {clave: 0 for clave in claves}

    for resultado in lista_resultados:
        for clave in claves:
            promedio[clave] += resultado[clave]

    for clave in claves:
        promedio[clave] /= len(lista_resultados)

    return promedio

def graficar_g2(promedio):
    eficiencias = promedio['eficiencias']
    dark_rates = promedio['dark_rates']
    g2 = promedio['g2']

    X, Y = np.meshgrid(dark_rates, eficiencias)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, g2, levels=50, cmap='viridis')
    plt.colorbar(cp, label=r'$g^{(2)}(0)$')
    plt.xlabel('Dark count rate [cuentas/ns]')
    plt.ylabel('Eficiencia cuántica')
    plt.title(r'Mapa de $g^{(2)}(0)$')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    resultados = cargar_resultados()
    promedio = promediar_resultados(resultados)
    graficar_g2(promedio)
