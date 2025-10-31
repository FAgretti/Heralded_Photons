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
    """Cuenta coincidencias temporales entre t1 y t2 usando operaciones vectorizadas"""
    t1 = t1[:, None]
    t2 = t2[None, :]
    delta = cp.abs(t1 - t2)
    coinciden = cp.any(delta <= ventana, axis=1)
    return int(cp.sum(coinciden))

def coincidencias_triples(t1, t2, t3, ventana):
    """Cuenta coincidencias triples de t1 con t2 y t3 usando operaciones vectorizadas"""
    t1_col = t1[:, None]
    
    delta12 = cp.abs(t1_col - t2[None, :])
    delta13 = cp.abs(t1_col - t3[None, :])

    c12 = cp.any(delta12 <= ventana, axis=1)
    c13 = cp.any(delta13 <= ventana, axis=1)

    n12 = int(cp.sum(c12))
    n13 = int(cp.sum(c13))
    n123 = int(cp.sum(c12 & c13))

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
n_bins = 1000
bin_duration = 1.0
ventana = 0.1
n_dark = 100  # total de cuentas oscuras por detector
duracion = n_bins * bin_duration
eff = 1

sim(n_bins,bin_duration,ventana,n_dark,duracion,eff)


# === Ejecución de la simulación ===
