import cupy as cp
import numpy as np
import pickle
from tqdm import tqdm

from utilidades import generar_pares, beam_splitter, filtro, detectar, coincidencias_triples

def simular(bins, it, tasa_pares=0.01, bin_duration=1.0, ventana=0.1):
    duracion = bins * bin_duration

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

    with open(f"resultados_simulacion_{it}.pkl", "wb") as f:
        pickle.dump(resultados, f)

if __name__ == "__main__":
    it = 20
    for i in range(it):
        simular(2000, i)

    print("Fin de simulacion.")
