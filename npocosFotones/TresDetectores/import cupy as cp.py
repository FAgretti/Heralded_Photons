import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from Dispositivos import BeamSplitter, Filtro, APD, coincidencias_temporales,coincidencias_triples


# Initialize parameters
Bins = 5000
Photon_Pair_Probability = 0.3  
Average_Photons_Per_Bin = 0.1  

VecFotones = []

# Generacion de fotones
for _ in range(Bins):
    # Numero de fotones por bin poissoneano
    num_photons = cp.random.poisson(Average_Photons_Per_Bin).item()

    if num_photons > 0:
        # Longitudes de onda en cada bin
        wavelengths = cp.random.choice([100, 300], size=num_photons, p=[0.5, 0.5])
        times = cp.random.randint(1, 1000, num_photons)
        
        # Si lo que genero es un par, agrego un foton mas
        if cp.random.rand() < Photon_Pair_Probability:
            wavelengths = cp.append(wavelengths, [100, 300])
            times = cp.append(times, [cp.random.randint(1, 1000), cp.random.randint(1, 1000)])
        
        VecFotones.append(cp.column_stack((times, wavelengths)))

# Paso a device
if VecFotones:
    VecFotones = cp.concatenate(VecFotones, axis=0)
else:
    VecFotones = cp.empty((0, 2), dtype=int)

# Rangos de eficiencia y cuentas oscuras
eficiencias = cp.linspace(1, 0.3, 5)
osc = cp.linspace(0, 10, 5)  

# Resultados
N12 = cp.zeros((len(eficiencias), len(osc)), dtype=int)
N13 = cp.zeros((len(eficiencias), len(osc)), dtype=int)
Bunch = cp.zeros((len(eficiencias), len(osc)), dtype=int)
N123 = cp.zeros((len(eficiencias), len(osc)), dtype=int)
N1 = cp.zeros((len(eficiencias), len(osc)), dtype=int)
g2 = cp.ones((len(eficiencias), len(osc)))

# BS y filtros
BS = BeamSplitter(0.5)
FiltroS = Filtro(F1=50, F2=200) 
FiltroI = Filtro(F1=150, F2=300) 

# Loop over efficiency levels for both detectors
for eff_idx, eff in enumerate(eficiencias):
    for k_idx, k in enumerate(osc):
        # Set dark count rate as a function of efficiency (can adjust formula)
        dark = 0.1 * k  # Adjusting dark count rate based on efficiency level
        detector = APD(eff, dark)
        
        # Count the number of detected photons in the first two columns of VecFotones
        nonzero_photons = cp.count_nonzero(VecFotones[:, :2], axis=1)
        #N1[eff_idx, k_idx] = cp.sum(nonzero_photons >= 1)
        #Bunch[eff_idx, k_idx] = cp.sum(nonzero_photons > 1)

        # Photon splitting using the beam splitter
        a, b = BS.Dividir(VecFotones[:, :2])
        
        # Apply filters to both paths
        camino1 = FiltroS.Filtrar(a)
        camino2 = FiltroI.Filtrar(b)
        
        # Detection results for the first two paths
        Res1 = detector.Detectar(camino1)
        Res2 = detector.Detectar(camino2)
        
        # Ensure Res1 and Res2 have the same size
        min_size = min(Res1.size, Res2.size)
        Res1 = Res1[:min_size]
        Res2 = Res2[:min_size]
        
        # Split camino2 into two paths
        c, d = BS.Dividir(camino2)
        
        # Detection results for paths c and d
        Res3 = detector.Detectar(c)
        Res4 = detector.Detectar(d)
        
        # Ensure Res3 and Res4 have the same size
        min_size = min(Res3.size, Res4.size)
        Res3 = Res3[:min_size]
        Res4 = Res4[:min_size]
        
        # Ensure all arrays have the same size for coincidence calculation
        min_size = min(Res1.size, Res2.size, Res3.size)
        Res1 = Res1[:min_size]
        Res2 = Res2[:min_size]
        Res3 = Res3[:min_size]
        
       # Extraer tiempos detectados (usando Res1 como máscara sobre camino1[:,0])
        t1 = camino1[:, 0][Res1]
        t2 = camino2[:, 0][Res2]
        t3 = c[:, 0][Res3]


        N1[eff_idx,k_idx] = coincidencias_temporales(t1,t1)
        N12[eff_idx, k_idx] = coincidencias_temporales(t1, t2)
        N13[eff_idx, k_idx] = coincidencias_temporales(t1, t3)
        N123[eff_idx, k_idx] = coincidencias_triples(t1, t2,t3)  # Para simplificar por ahora
        


        # Ensure normalization for g2 computation: prevent overflow
        if N12[eff_idx, k_idx] > 0 and N13[eff_idx, k_idx] > 0:
            g2[eff_idx, k_idx] = N1[eff_idx, k_idx]*N123[eff_idx, k_idx] / (N12[eff_idx, k_idx] * N13[eff_idx, k_idx])  # Correct normalization of g2

# Convert results back to numpy for plotting
eficiencias = cp.asnumpy(eficiencias)
osc = cp.asnumpy(osc)
g2 = cp.asnumpy(g2)
g2 = g2.T

print(np.shape(g2))

# Plot results
plt.imshow((g2), aspect='auto', cmap='viridis', origin='lower', extent=[eficiencias[0], eficiencias[-1], osc[0], osc[-1]])
plt.colorbar(label="g2")
plt.xlabel("Eficiencia")
plt.ylabel("Cuentas oscuras")
plt.title("g2 ")
plt.show()


