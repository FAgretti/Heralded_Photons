import cupy as cp

def generar_fotones(n_bins, bin_duration):
    """Genera un par de fotones por bin con timestamps aleatorios dentro del bin"""
    t_base = cp.arange(n_bins) * bin_duration
    jitter = cp.random.uniform(0, bin_duration, size=(n_bins, 1))
    tiempos = t_base[:, None] + jitter  # shape (n_bins, 1)
    longitudes = cp.array([[100, 300]])  # mismo para todos los bins
    tiempos = cp.tile(tiempos, (1, 2)).reshape(-1, 1)
    longitudes = cp.tile(longitudes, (n_bins, 1)).reshape(-1, 1)
    return cp.concatenate([tiempos, longitudes], axis=1)

def split_y_filtrar(fotones):
    """Splitter y filtros fijos"""
    # Beam splitter inicial
    mask_bs = cp.random.rand(len(fotones)) < 0.5
    a, b = fotones[mask_bs], fotones[~mask_bs]

    # Filtros de longitud de onda
    s = a[(a[:,1] > 50) & (a[:,1] < 200)]
    i = b[(b[:,1] > 200) & (b[:,1] < 400)]

    s = a
    i = b
    # Segundo beam splitter
    mask_bs2 = cp.random.rand(len(i)) < 0.5
    c, d = i[mask_bs2], i[~mask_bs2]

    return s[:, 0], c[:, 0], d[:, 0]

def agregar_cuentas_oscuras(t, n_dark, duracion):
    """Agrega cuentas oscuras uniformemente distribuidas"""
    darks = cp.random.uniform(0, duracion, int(n_dark))
    return cp.sort(cp.concatenate([t, darks]))

def coincidencias(t1, t2, ventana):
    """Cuenta coincidencias temporales entre t1 y t2 usando algoritmo eficiente."""
    t1 = cp.sort(t1)
    t2 = cp.sort(t2)
    i, j = 0, 0
    count = 0
    while i < len(t1) and j < len(t2):
        dt = t1[i] - t2[j]
        if cp.abs(dt) <= ventana:
            count += 1
            i += 1
            j += 1
        elif dt < -ventana:
            i += 1
        else:
            j += 1
    return count

def coincidencias_triples(t1, t2, t3, ventana):
    t1 = cp.sort(t1)
    t2 = cp.sort(t2)
    t3 = cp.sort(t3)
    n12 = coincidencias(t1, t2, ventana)
    n13 = coincidencias(t1, t3, ventana)
    n123 = 0
    # For each t1, find t2 and t3 within window, then count only one triple per t1
    for i in range(len(t1)):
        j = cp.searchsorted(t2, t1[i] - ventana, side='left')
        k = cp.searchsorted(t3, t1[i] - ventana, side='left')
        found = False
        while j < len(t2) and t2[j] <= t1[i] + ventana and not found:
            while k < len(t3) and t3[k] <= t1[i] + ventana and not found:
                if cp.abs(t2[j] - t3[k]) <= ventana:
                    n123 += 1
                    found = True  # Only count one triple per t1
                k += 1
            j += 1
    return n12, n13, n123

def detectar(eff,t):
    detection_probabilities = cp.random.rand(t.shape[0])
    detectados = detection_probabilities < eff
    tiempos_fotones = t[:][detectados]
    return tiempos_fotones


def sim(n_bins,bin_duration,ventana,n_dark,duracion,eff):
    fotones = generar_fotones(n_bins, bin_duration)
    t1, t2, t3 = split_y_filtrar(fotones)
    t1 = detectar(eff,t1)
    t2 = detectar(eff,t2)
    t3 = detectar(eff,t3)

    t1 = agregar_cuentas_oscuras(t1, n_dark, duracion)
    t2 = agregar_cuentas_oscuras(t2, n_dark, duracion)
    t3 = agregar_cuentas_oscuras(t3, n_dark, duracion)

    n12, n13, n123 = coincidencias_triples(t1, t2, t3, ventana)
    N1 = len(cp.unique(t1))

    # === Cálculo de g2 ===
    if n12 > 0 and n13 > 0:
        g2 = (N1 * n123) / (n12 * n13)
    else:
        g2 = 0

    print(f"N1 = {N1}, N12 = {n12}, N13 = {n13}, N123 = {n123}")
    print(f"g2 = {g2:.4f}")
    return g2


# === Parámetros de la simulación ===
if __name__ == "__main__":
    n_bins = 1000
    bin_duration = 1.0
    ventana = 0.1
    n_dark = 100  # total de cuentas oscuras por detector
    duracion = n_bins * bin_duration
    eff = 1

    sim(n_bins,bin_duration,ventana,n_dark,duracion,eff)


# === Ejecución de la simulación ===
