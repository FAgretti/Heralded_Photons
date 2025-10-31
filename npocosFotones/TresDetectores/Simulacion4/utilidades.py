# utilidades.py

import cupy as cp
import numpy as np
from numba import njit

# ------------------ Generador de pares de fotones ------------------
def generar_pares(tasa_pares, bins, bin_duration):
    n_pares = cp.random.poisson(tasa_pares, size=bins)
    total_pares = int(cp.sum(n_pares))

    if total_pares == 0:
        return cp.empty((0, 2))

    tiempos = cp.random.uniform(0, bin_duration, size=(total_pares,))
    #tiempos += cp.repeat(cp.arange(bins) * bin_duration, )

    longitudes = cp.tile(cp.array([100, 300]), total_pares)
    tiempos = cp.repeat(tiempos, 2)

    return cp.stack((tiempos, longitudes), axis=1)

# ------------------ Beam Splitter ------------------
def beam_splitter(fotones):
    mask = cp.random.rand(len(fotones)) < 0.5
    return fotones[mask], fotones[~mask]

# ------------------ Filtro de longitudes de onda ------------------
def filtro(fotones, fmin, fmax):
    longitudes = fotones[:, 1]
    mask = (longitudes >= fmin) & (longitudes <= fmax)
    return fotones[mask]

# ------------------ Detector con eficiencia y dark counts ------------------
def detectar(fotones, eficiencia, dark_rate, duracion):
    mask = cp.random.rand(len(fotones)) < eficiencia
    tiempos_detectados = fotones[mask, 0]
    n_dark = cp.random.poisson(dark_rate * duracion)
    tiempos_dark = cp.random.uniform(0, duracion, size=int(n_dark))
    return cp.sort(cp.concatenate((tiempos_detectados, tiempos_dark)))

# ------------------ Coincidencias dobles ------------------
#@njit
def coincidencias(t1, t2, ventana):
    coincidencias = 0
    i = 0
    j = 0
    while i < len(t1) and j < len(t2):
        if abs(t1[i] - t2[j]) <= ventana:
            coincidencias += 1
            i += 1
            j += 1
        elif t1[i] < t2[j]:
            i += 1
        else:
            j += 1
    return coincidencias

# ------------------ Coincidencias triples ------------------
#@njit
def coincidencias_triples(t1, t2, t3, ventana):
    n12 = coincidencias(t1, t2, ventana)
    n13 = coincidencias(t1, t3, ventana)

    n123 = 0
    i, j, k = 0, 0, 0
    while i < len(t1):
        while j < len(t2) and t2[j] < t1[i] - ventana:
            j += 1
        while k < len(t3) and t3[k] < t1[i] - ventana:
            k += 1
        if j < len(t2) and k < len(t3):
            if abs(t1[i] - t2[j]) <= ventana and abs(t1[i] - t3[k]) <= ventana:
                n123 += 1
        i += 1
    return n12, n13, n123