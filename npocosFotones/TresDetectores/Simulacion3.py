# --- Librerias ---
import cupy as cp
import numpy as np
import pickle
from tqdm import tqdm
import numba
from numba import njit, prange

# --- Funciones principales ---

def generar_pares(tasa_pares, bins, bin_duration):
    VecFotones = []
    for i in range(bins):
        n_pares = cp.random.poisson(tasa_pares)
        if n_pares > 0:
            tiempos = bin_duration * i + cp.random.uniform(0, bin_duration, size=(int(n_pares), 1))
            longitudes = cp.tile(cp.array([[100], [300]]), (int(n_pares), 1))
            tiempos = cp.tile(tiempos, (1, 2)).reshape(-1, 1)
            VecFotones.append(cp.concatenate((tiempos, longitudes), axis=1))
    return cp.concatenate(VecFotones, axis=0) if VecFotones else cp.empty((0, 2))

def beam_splitter(fotones):
    mask = cp.random.rand(len(fotones)) < 0.5
    return fotones[mask], fotones[~mask]

def filtro(fotones, f1, f2):
    longitudes = fotones[:, 1]
    mask = (longitudes >= f1) & (longitudes <= f2)
    return fotones[mask]

def detectar(fotones, eficiencia, dark_rate, duracion):
    mask = cp.random.rand(len(fotones)) < eficiencia
    tiempos_detectados = fotones[mask, 0]
    n_dark = cp.random.poisson(dark_rate * duracion)
    tiempos_dark = cp.random.uniform(0, duracion, size=int(n_dark))
    return cp.sort(cp.concatenate((tiempos_detectados, tiempos_dark)))

# --- Coincidencias normales optimizado GPU ---
@cp.fuse()
def coincidencias(t1, t2, ventana):
    i, j = 0, 0
    coincidencias = 0
    while i < len(t1) and j < len(t2):
        dt = t1[i] - t2[j]
        if cp.abs(dt) <= ventana:
            coincidencias += 1
            i += 1
            j += 1
        elif dt < -ventana:
            i += 1
        else:
            j += 1
    return coincidencias

# --- Coincidencias triples CPU super acelerado ---
@njit(parallel=True)
def coincidencias_triples_CPU(t1, t2, t3, ventana):
    n12 = 0
    n13 = 0
    n1 = 0 
    n123 = 0

    for i in prange(len(t1)):
        n1 +=1
        for j in range(len(t2)):
            if np.abs(t1[i] - t2[j]) <= ventana:
                n12 += 1
        for k in range(len(t3)):
            if np.abs(t1[i] - t3[k]) <= ventana:
                n13 += 1
            if (np.abs(t2[j] - t3[k]) <= ventana and np.abs(t1[i] - t2[j]) <= ventana and np.abs(t1[i] - t3[k]) <= ventana):
                n123 += 1
    return n1, n12, n13, n123

# --- Funcion de simulacion ---
def simular(bins_, it):
    bins = bins_
    bin_duration = 1.0
    duracion = bins * bin_duration
    tasa_pares = 0.01
    ventana = 0.1

    eficiencias = cp.linspace(1, 0.5, 10)
    dark_rates = cp.linspace(0.0, 1, 10)

    g2 = np.zeros((len(eficiencias), len(dark_rates)))
    N12 = np.zeros_like(g2)
    N13 = np.zeros_like(g2)
    N123 = np.zeros_like(g2)
    N1 = np.zeros_like(g2)

    VecFotones = generar_pares(tasa_pares, bins, bin_duration)

    for i, eff in enumerate(tqdm(eficiencias.get(), desc="Barrido eficiencia")):
        for j, dark in enumerate(dark_rates.get()):
            a, b = beam_splitter(VecFotones)
            s = filtro(a, 50, 200)
            i_ = filtro(b, 150, 400)
            c, d = beam_splitter(i_)

            t1 = detectar(s, eff, dark, duracion).get()
            t2 = detectar(c, eff, dark, duracion).get()
            t3 = detectar(d, eff, dark, duracion).get()

            t1.sort()
            t2.sort()
            t3.sort()

            n1, n12, n13, n123 = coincidencias_triples_CPU(t1, t2, t3, ventana)

            N12[i, j] = n12
            N13[i, j] = n13
            N123[i, j] = n123
            N1[i, j] = n1

            if n12 > 0 and n13 > 0:
                g2[i, j] = N1[i, j] * n123 / (n12 * n13)

    resultados = {
        'eficiencias': eficiencias.get(),
        'dark_rates': dark_rates.get(),
        'g2': g2,
        'N12': N12,
        'N13': N13,
        'N123': N123,
        'N1': N1
    }

    print(N1)
    print(N12)
    print(N13)
    print(N123)

    with open(f"TresDetectores/Res/resultados_simulacion_{it}.pkl", "wb") as f:
        pickle.dump(resultados, f)

# --- Main ---
if __name__ == "__main__":
    it = 10
    for i in range(it):
        simular(10000, i)
    print("Fin de simulaciones")