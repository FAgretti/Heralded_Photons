# Simulacion.py

import cupy as cp
import Disp as disp

def generar_pares(tasa_pares, duracion_total):
    """
    Genera tiempos de emisión de pares de fotones.
    """
    #n_pares = cp.random.poisson(tasa_pares * duracion_total)
    n_pares = tasa_pares * duracion_total
    tiempos = cp.random.uniform(0, duracion_total, int(n_pares))
    return tiempos

def coincidencias(t1, t2, ventana):
    """Cuenta coincidencias temporales entre t1 y t2"""
    count = 0
    for t in t1:
        c2 = cp.any(cp.abs(t - t2) <= ventana)
        if c2:
            count += 1
    return count

def coincidencias_triples(t1, t2, t3, ventana):
    """Cuenta coincidencias triples de t1 con t2 y t3 dentro de la ventana"""
    n12 = coincidencias(t1, t2, ventana)
    n13 = coincidencias(t1, t3, ventana)
    n123 = 0
    for t in t1:
        c2 = cp.any(cp.abs(t - t2) <= ventana)
        c3 = cp.any(cp.abs(t - t3) <= ventana)
        if c2 and c3:
            n123 += 1
    return n12, n13, n123


def medir_g2(tasa_pares, eficiencia, dark_count, duracion_total, ventana):
    """
    Ejecuta toda la simulación y calcula g2(0).
    """
    # Generar pares
    tiempos_pares = generar_pares(tasa_pares, duracion_total)

    # Camino 1 y 2: primero un beam splitter
    BS = disp.BeamSplitter(R=0.5)
    camino1, camino2 = BS.dividir(tiempos_pares)

    # Camino 2: segundo beam splitter
    c, d = BS.dividir(camino2)

    # Detectores
    APD1 = disp.APD(eficiencia, dark_count)
    APD2 = disp.APD(eficiencia, dark_count)
    APD3 = disp.APD(eficiencia, dark_count)

    # Detección
    t1 = APD1.detectar(camino1, duracion_total)
    t2 = APD2.detectar(c, duracion_total)
    t3 = APD3.detectar(d, duracion_total)

    # Coincidencias
    #n12 = coincidencias(t1, t2, ventana)
    #n13 = coincidencias(t1, t3, ventana)
    n12, n13, n123 = coincidencias_triples(t1, t2, t3, ventana)
    print(n123)
    n1 = len(t1)

    # g2(0)
    if (n12 + n13) == 0:
        g2 = 0
    else:
        g2 = (n1*n123) / (n12 + n13)

    return g2

# ==== Parámetros de simulación ====
if __name__ == "__main__":
    tasa_pares = int(100000)     # 100000 pares por segundo
    duracion_total = int(1)      # 1 segundo
    ventana = int(1e-1)        # 1 ns de ventana de coincidencia

    eficiencia = 0.5
    dark_count = 1000

    g2 = medir_g2(tasa_pares, eficiencia, dark_count, duracion_total, ventana)

    print(f"g2(0) = {g2:.4f}")
if __name__ == "__main__":
    tasa_pares = int(100000)     # 100000 pares por segundo
    duracion_total = int(1)      # 1 segundo
    ventana = 1e-9               # 1 ns de ventana de coincidencia

    eficiencia = 0.5
    dark_count = 1000

    g2 = medir_g2(tasa_pares, eficiencia, dark_count, duracion_total, ventana)

    print(f"g2(0) = {g2:.4f}")
