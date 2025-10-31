import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import cupy as cp


"""
Defino los dispositivos que estarán presentes durante el experimento
APD:
    Qef: Quantum efficiency
    Dark: Dark counts
    Temp: Temperature
    BW: Bandwidth


    Detectar(fotones): Método que simula la detección de fotones
    Ruido: Método que simula el ruido de un APD   

    
SNSPD:
    Qef: Quantum efficiency
    Dark: Dark counts
    Temp: Temperature
    BW: Bandwidth
    Tc: Critical temperature
    Rn: Normal resistance
    Rsh: Shunt resistance
    Ic: Critical current
    F: Fano factor
    R: Resistencia

    Detectar(fotones): Método que simula la detección de fotones
    Ruido: Método que simula el ruido de un SNSPD

BeamSplitter:
    T: Transmission
    R: Reflection

    Dividir(fotones): Método que simula la división de fotones a partir de un vector que contiene los muestreos temporales

Filtro:
    F1: Frecuencia mínima, por defecto 0
    F2: Frecuencia máxima, por defecto inf
    Si se desea un filtro pasa bajos solo se le pasa un argumento a F2
    Si se desea un filtro pasa altos solo se le pasa un argumento a F1
    Si se desea un filtro pasa banda se le pasan ambos argumentos

Durante el experimentos los datos se moveran en forma de un vector de fotones, donde cada elemento del vector corresponde a un muestreo temporal y contiene la cantidad de fotones detectados en ese instante y la longitud de onda a la que pertenecen.
"""

""" class APD:
    def __init__(self, Qef, Dark):
        self.Qef = Qef
        self.Dark = Dark

    def Detectar(self, fotones):
        fotones_detectados = cp.zeros(len(fotones))
        fotones_detectados = cp.where(cp.random.rand(len(fotones)) < self.Qef, fotones, 0)
        cuentas_oscuras = cp.random.poisson(self.Dark, len(fotones))
        fotones_detectados += cuentas_oscuras
        return fotones_detectados """

class APD:
    def __init__(self, Qef, Dark):
        # Validación de entradas
        if not (0 <= Qef <= 1):
            raise ValueError("Quantum efficiency Qef must be between 0 and 1")
        if Dark < 0:
            raise ValueError("Dark count rate Dark must be non-negative")

        self.Qef = Qef  # Eficiencia cuántica
        self.Dark = Dark  # Tasa de cuentas oscuras (eventos por unidad de tiempo)

    def Detectar(self, fotones, duracion_total):
        if fotones.shape[0] == 0:
            tiempos_fotones = cp.array([], dtype=int)
        else:
            detection_probabilities = cp.random.rand(fotones.shape[0])
            detectados = detection_probabilities < self.Qef
            tiempos_fotones = fotones[:, 0][detectados]

        # Cuentas oscuras simuladas como eventos aleatorios independientes
        num_dark = cp.random.poisson(self.Dark * duracion_total)
        tiempos_totales = agregar_cuentas_oscuras(tiempos_fotones, num_dark, duracion_total)


        
        # Mezclar tiempos reales y oscuros
        #tiempos_totales = cp.concatenate((tiempos_fotones, tiempos_dark))

        return cp.sort(tiempos_totales)

def agregar_cuentas_oscuras(t, n_dark, duracion):
    """Agrega cuentas oscuras uniformemente distribuidas"""
    darks = cp.random.uniform(0, duracion, size=n_dark)
    return cp.sort(cp.concatenate([t, darks]))

 
class BeamSplitter:
    def __init__(self, T):
        self.T = T  # Transmission probability

    def Dividir(self, fotones):

        fotonesT = cp.zeros_like(fotones)  
        fotonesR = cp.zeros_like(fotones)  

        rnd_vals = cp.random.random(fotones.shape[0])  
        fotonesT[rnd_vals < self.T] = fotones[rnd_vals < self.T]
        fotonesR[rnd_vals >= self.T] = fotones[rnd_vals >= self.T]
        
        return fotonesT, fotonesR
        
class Filtro:
    def __init__(self, F1 = 0, F2 = cp.inf):
        self.F1 = F1
        self.F2 = F2

    def Filtrar(self, fotones):
        return cp.where(cp.logical_and(fotones[:] > self.F1, fotones[:] < self.F2), fotones, 0)
    

def coincidencias_temporales(t1, t2, ventana=10):
    t1 = cp.sort(t1[t1 > 0])
    t2 = cp.sort(t2[t2 > 0])
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

def coincidencias_dobles_y_triples(t1, t2, t3, ventana=10):
    """
    Devuelve las coincidencias dobles (N12, N13) y las triples (N123).
    Garantiza que N123 ≤ min(N12, N13), como debe ser físicamente.
    """
    t1 = cp.sort(t1[t1 > 0])
    t2 = cp.sort(t2[t2 > 0])
    t3 = cp.sort(t3[t3 > 0])

    n12 = 0
    n13 = 0
    n123 = 0

    j2 = 0
    j3 = 0

    for i in range(len(t1)):
        # Coincidencias con t2
        while j2 < len(t2) and t2[j2] < t1[i] - ventana:
            j2 += 1
        k2 = j2
        c2 = -1
        while k2 < len(t2) and t2[k2] <= t1[i] + ventana:
            if abs(t1[i] - t2[k2]) <= ventana:
                c2 = t2[k2]
                n12 += 1
                break  # solo una coincidencia por t1

            k2 += 1

        # Coincidencias con t3
        while j3 < len(t3) and t3[j3] < t1[i] - ventana:
            j3 += 1
        k3 = j3
        c3 = -1
        while k3 < len(t3) and t3[k3] <= t1[i] + ventana:
            if abs(t1[i] - t3[k3]) <= ventana:
                c3 = t3[k3]
                n13 += 1
                break  # solo una coincidencia por t1

            k3 += 1

        # Si hay coincidencia con t2 y t3, verificamos también t2 vs t3
        if c2 != -1 and c3 != -1 and abs(c2 - c3) <= ventana:
            n123 += 1

    return n12, n13, n123






