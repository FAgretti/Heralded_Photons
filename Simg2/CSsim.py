#la fuente tienen una distribucion poissoniana, genera pares de fotones con una tasa media de 0.1 por unidad de tiempo
import numpy as np
import matplotlib.pyplot as plt

rate = 0.3
time = 500000
windows = np.linspace(10, 200, 200, dtype=int)
photons1 = np.random.poisson(rate, time)


photons2 = photons1.copy()


#beam splitter 50/50
idler_in = np.array([np.random.binomial(i, 0.5) for i in photons1])
signal_in = np.array([np.random.binomial(i, 0.5) for i in photons2])

g2fuente = np.mean(idler_in*(photons1-idler_in))/np.mean(photons1*0.5)**2

g2ii_list = []
g2ss_list = []
g2is_list = []

for j in range(1,len(windows)+1):
    window = windows[j-1]
    # print("Window size: ", window)
    idler_split1 = np.zeros(time//window)
    idler_split1 = np.array([np.random.binomial(i, 0.5) for i in idler_in])
    idler_split2 = idler_in - idler_split1

    idler_out = np.zeros(time//window)
    for i in range(time//window): idler_out[i] = np.sum(idler_split1[i*window:(i+1)*window])
    idler_out2 = np.zeros(time//window)
    for i in range(time//window): idler_out2[i] = np.sum(idler_split2[i*window:(i+1)*window])

    signalsplit1 = np.array([np.random.binomial(i, 0.5) for i in signal_in])
    signalsplit2 = signal_in - signalsplit1

    signal_out = np.zeros(time//window)
    for i in range(time//window): signal_out[i] = np.sum(signalsplit1[i*window:(i+1)*window])
    signal_out2 = np.zeros(time//window)
    for i in range(time//window): signal_out2[i] = np.sum(signalsplit2[i*window:(i+1)*window])

    #correlacion entre idler y signal
    g2is= np.mean((idler_out+idler_out2)*(signal_out+signal_out2))/ (np.mean(idler_out+idler_out2)*np.mean(signal_out+signal_out2))
    g2ii= np.mean(idler_out*idler_out2)/ (np.mean(idler_out)*np.mean(idler_out2))
    g2ss= np.mean(signal_out*signal_out2)/ (np.mean(signal_out)*np.mean(signal_out2))
    g2is_list.append(g2is)
    g2ii_list.append(g2ii)
    g2ss_list.append(g2ss)

# print("g2is: ", g2is_list)
# print("g2ii: ", g2ii_list)
# print("g2ss: ", g2ss_list)

plt.figure()
plt.plot(windows, g2is_list, label=r'$g^{(2)}_{is}$', linewidth=3)
plt.plot(windows, g2ii_list, label=r'$g^{(2)}_{ii}$', linewidth=3)
plt.plot(windows, g2ss_list, label=r'$g^{(2)}_{ss}$', linewidth=3)
print("g2fuente: ", g2fuente)
#plt.plot(windows, (1/(np.mean(photons1)*windows))+1, 'r--', label=r'$g^{(2)}_{is}$ teórico', linewidth=3)
#plt.plot(windows, (1/(rate*windows))+1, 'k--')

plt.xlabel('Ventana de integración')
plt.ylabel(r'$g^{(2)}(0)$')


plt.title('g2 vs tamaño de ventana', fontsize=20)
plt.grid()
plt.xlabel('Fotones por ventana', fontsize=18)
plt.ylabel(r'$g^{(2)}(0)$', fontsize=18)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize = 18)

plt.show()

# plt.figure()
# plt.loglog(windows, g2is_list-np.ones(200), label=r'$$g^{(2)}_{is} - 1$$')
