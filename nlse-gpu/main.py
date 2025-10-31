
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import Telecotoolbox as ttb
from sim import Sim, Fibra
import solver
import scipy.constants as const

#---Parámetros de la simulación---
puntos = 2**13
Tmax   = 1000 #ps
sim = Sim(puntos, Tmax)
fs = 1/sim.paso_t

#---Parámetros de la fibra---
L = 4*8976 #m
gamma = 1.4e-3 #1/(W*m)
gamma1 = 0
alpha = 0
lambda0 = 1550 #nm
betas = [-20e-3,0*10e-3,0*1e-3] #ps^n/m
fib = Fibra(L, gamma, gamma1, alpha, lambda0, betas)


#---Pulso de entrada---
Pot = 0.25 #W
#Para 8 W se obtiene longitud de coeherencia inf

#---Parametros del FWM
freqShift = cp.sqrt(gamma*Pot/p.abs(betas[0])) #THz
#freqShift = 0.23*(2*cp.pi)
freqFWM = freqShift/(2*cp.pi)

#pulso_0 = 1.5*cp.sqrt(Pot)*cp.ones(puntos)*cp.exp(-1j*2*cp.pi*freqShift*sim.tiempo)
pump = 1*cp.sqrt(Pot)*cp.ones(puntos)*cp.exp(-1j*2*cp.pi*0*freqShift*sim.tiempo)
pulso_1 = 0.1*cp.sqrt(Pot)*cp.ones(puntos)*cp.exp(1j*2*cp.pi*freqFWM*sim.tiempo)
ruido = 0.01*cp.sqrt(Pot)*cp.random.normal(0,0.1,puntos)

pulso_0 = pump +pulso_1+ ruido

#---Simulación---
z_locs = 100
z, AW, AT = solver.SolveNLS(sim, fib, pulso_0, z_locs=z_locs)

deltaKm = betas[0]*(freqShift)**2+(betas[2]/12)*(freqShift)**4
deltaKnl = gamma*Pot

kappa = deltaKm + deltaKnl
if(kappa):
    Lcoh = cp.inf
else:
    Lcoh = (2*cp.pi/kappa) #m?
print("Longitud de coherencia = ",Lcoh)

Linteraccion = (cp.pi/(gamma*Pot))
print("Longitud de interaccion = ",Linteraccion)

#La longitud en la que se da la conversion total de potencia del pump al idler y signal sigue una ley de 2*pi/(2*gamma*Pot)
#Para 16W ~ 150m
#Para 8W ~ 300m         #Todos con gamma  = 1.4 e-3
#Para 4W ~ 600m

print("Frecuencia = ",freqFWM)
#---Gráficos---
plt.figure()

plt.subplot(211)
"""
plt.plot(sim.tiempo, cp.abs(pulso_0)**2)

plt.title("Pulso de entrada", fontsize=24)
plt.subplot(212)
"""
plt.plot(sim.freq, cp.abs(ttb.TFoptGPU(pulso_0)**2))

plt.title("Espectro de entrada", fontsize=16)
plt.xlabel("Frecuencia [THz]", fontsize=16)
plt.ylabel("Potencia [W]", fontsize=16)
plt.xlim(-0.5,0.5)

#plt.figure()
"""
plt.subplot(211)
plt.plot(sim.tiempo, cp.abs(AT[-1])**2)
plt.title("Pulso de salida",fontsize=24)
"""
plt.subplot(212)

plt.plot(sim.freq, cp.abs(AW[-1])**2)
plt.title("Espectro de salida",fontsize=16)
plt.xlabel("Frecuencia [THz]", fontsize=16)
plt.ylabel("Potencia [W]", fontsize=16)
plt.xlim(-0.5,0.5)

plt.subplots_adjust(left=0.1, right=0.9, 
                    top=0.9, bottom=0.1, 
                    wspace=0.4, hspace=0.6)

plt.figure()
#plt.imshow(cp.abs(cp.fft.fftshift(ttb.TFopt(AT)))**2, aspect='auto', extent=[cp.min(sim.freq), cp.max(sim.freq),0,L])
AW = cp.sqrt(Pot)*(cp.fft.fftshift(AW,axes=1)/cp.max(cp.abs(AW)))
plt.imshow((cp.abs(AW)**2), aspect='auto', extent=[cp.min(sim.freq), cp.max(sim.freq),0,L], origin='lower')
plt.xlim(cp.min(sim.freq), cp.max(sim.freq))
#plt.xlim(-2,2)
plt.colorbar()
plt.title("Espectro de salida en función de z")

plt.figure()
plt.imshow(cp.abs(AT)**2, aspect='auto', origin='lower', extent=[sim.tiempo[0], sim.tiempo[-1],0,L])
#plt.ylim([0, Tmax])
plt.xlim([-20, 20])
plt.colorbar()
plt.title("Evolución temporal del pulso")

plt.figure()
ind = cp.where(cp.abs(AW[0,:])>0.09*cp.sqrt(Pot))
print(ind[0])
plt.plot(cp.linspace(0,L,z_locs),cp.abs(AW[:,ind[0][0]])**2, label = "Idler")
plt.plot(cp.linspace(0,L,z_locs),cp.abs(AW[:,ind[0][1]])**2, label = "Pump")
plt.plot(cp.linspace(0,L,z_locs),cp.abs(AW[:,(2*ind[0][1]-ind[0][0])])**2, label = "Signal")
plt.plot(cp.linspace(0,L,z_locs),cp.abs(AW[:,(2*ind[0][1]-ind[0][0])])**2 + cp.abs(AW[:,ind[0][0]])**2, label = "Signal + idler")
#plt.plot(cp.linspace(0,L,z_locs),cp.abs(AW[:,4096])**2, label = "pump")
suma = cp.zeros(z_locs)
for i in range(len(ind[0])):
    suma = suma + cp.abs(AW[:,ind[0][i]])**2
suma = suma - cp.abs(AW[:,4096])**2
#plt.plot(cp.linspace(0,L,z_locs),suma, label = "Productos no lineales")
plt.grid()
plt.legend(loc = "upper right", fontsize=16)
#vector del FWM
"""
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = '\n'.join((
    r'$\Delta k_{m} = %.3e$ $[\frac{1}{m}]$' % (deltaKm, ),
    r'$\Delta k_{nl} = %.3e$ $[\frac{1}{m}]$' % (deltaKnl, ),
    r'$\kappa = %.3e$ $[\frac{1}{m}]$' % (kappa, ),
    r'$L_{coh} = %.3e$ [m]' % (Lcoh, ),
    r'$L_{interaccion} = %.3e$ [m]' % (Linteraccion, )))


plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
"""
plt.title("Potencia del FWM en función de z", fontsize=20)
plt.xlabel("z [m]", fontsize=20)
plt.ylabel("Potencia [W]", fontsize=20)

plt.show()

# %%
