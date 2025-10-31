import cupy as cp
import numpy as np
import pickle
from tqdm import tqdm

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

def coincidencias(t1, t2, ventana):
    i, j = 0, 0
    coincidencias = 0
    while i < len(t1) and j < len(t2):
        dt = t1[i] - t2[j]
        if abs(dt) <= ventana:
            coincidencias += 1
            i += 1
            j += 1
        elif dt < -ventana:
            i += 1
        else:
            j += 1
    return coincidencias

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

def simular(bins_,it):
    bins = bins_
    bin_duration = 1.0
    duracion = bins * bin_duration
    tasa_pares = 0.01
    ventana = 0.1

    eficiencias = cp.linspace(0.5, 0.1, 10)
    dark_rates = cp.linspace(0.0, 0.3, 10)

    g2 = cp.zeros((len(eficiencias), len(dark_rates)))
    N12 = cp.zeros_like(g2)
    N13 = cp.zeros_like(g2)
    N123 = cp.zeros_like(g2)
    N1 = cp.zeros_like(g2)

    VecFotones = generar_pares(tasa_pares, bins, bin_duration)

    for i, eff in enumerate(tqdm(eficiencias.get(), desc="Barrido eficiencia")):
        for j, dark in enumerate(dark_rates):
            a, b = beam_splitter(VecFotones)
            s = filtro(a, 50, 200)
            i_ = filtro(b, 150, 300)
            c, d = beam_splitter(i_)

            t1 = detectar(s, eff, dark, duracion)
            t2 = detectar(c, eff, dark, duracion)
            t3 = detectar(d, eff, dark, duracion)

            n12, n13, n123 = coincidencias_triples(t1, t2, t3, ventana)

            N12[i, j] = n12
            N13[i, j] = n13
            N123[i, j] = n123
            N1[i, j] = len(cp.unique(t1))

            if n12 > 0 and n13 > 0:
                g2[i, j] = N1[i, j] * n123 / (n12 * n13)

    resultados = {
        'eficiencias': eficiencias.get(),
        'dark_rates': dark_rates.get(),
        'g2': g2.get(),
        'N12': N12.get(),
        'N13': N13.get(),
        'N123': N123.get(),
        'N1': N1.get()
    }

    with open("resultados_simulacion_%i.pkl" %(it), "wb") as f:
        pickle.dump(resultados, f)

if __name__ == "__main__":
    it = 20
    for i in range(it):
        simular(8000,i)

    print("Fin")
