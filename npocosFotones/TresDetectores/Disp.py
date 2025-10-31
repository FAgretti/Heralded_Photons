# Dispositivos.py

import cupy as cp

class BeamSplitter:
    def __init__(self, R=0.5):
        self.R = R  # Coeficiente de reflexión

    def dividir(self, fotones):
        """
        Divide los fotones en dos caminos de acuerdo al coeficiente de reflexión.
        """
        decisiones = cp.random.random(len(fotones))
        transmitidos = fotones[decisiones > self.R]
        reflejados = fotones[decisiones <= self.R]
        return transmitidos, reflejados

class APD:
    def __init__(self, eficiencia=1.0, dark_rate=0.0):
        self.eficiencia = eficiencia
        self.dark_rate = dark_rate  # dark_rate en cuentas por segundo

    def detectar(self, fotones, duracion_total):
        """
        Detecta los fotones con cierta eficiencia y agrega cuentas oscuras.
        """
        # Detectar con eficiencia
        decisiones = cp.random.random(len(fotones))
        fotones_detectados = fotones[decisiones < self.eficiencia]

        # Cuentas oscuras
        n_dark = cp.random.poisson(self.dark_rate * duracion_total)
        tiempos_dark = cp.random.uniform(0, duracion_total, int(n_dark))

        # Juntar todo
        eventos_totales = cp.concatenate([fotones_detectados, tiempos_dark])

        # Ordenar eventos
        eventos_totales = cp.sort(eventos_totales)

        return eventos_totales
