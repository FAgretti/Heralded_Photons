
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

#para cada bin temporal tengo una distribucion de probabilidad poissoneana
numbins = 10000
photons = np.tile(1,numbins) #np.random.poisson(lam=1, size=numbins)
#photons = np.random.poisson(lam=1, size=numbins)
photons2 = np.tile(1,numbins) #np.random.poisson(lam=1, size=numbins)
#photons2 = np.random.poisson(lam=1, size=numbins)
det_input = np.array([np.random.binomial(i, 0.5) for i in photons])
windows = np.linspace(1,100,100,dtype=int)   
dark = np.linspace(0,1,100)
g2 = np.zeros(100)
for j in range(100):
    window = int(windows[j])
    #window = 1
    print(f"Window size: {window}")
    det_output = np.zeros(numbins//window)
    i=0
    for i in range(numbins//window): det_output[i] = np.sum(det_input[i*window:(i+1)*window])

    # print(photons)
    # print(det_input)
    # print(det_output)

    # mean_photons = np.mean(photons)
    # mean_detinput = np.mean(det_input)
    # mean_detoutput = np.mean(det_output)/window

    # var_photons = np.var(photons)
    # var_detinput = np.var(det_input)
    # var_detoutput = np.var(det_output)/window


    # print(f"Mean photons: {mean_photons}, Variance photons: {var_photons}")
    # print(f"Mean det_input: {mean_detinput}, Variance det_input: {var_detinput}")
    # print(f"Mean det_output: {mean_detoutput}, Variance det_output: {var_detoutput}")

    #simulacion ruido
    #El detector introduce ruido correspondiente a la corriente de oscuridad. Este tiene una distribucion poissoneana https://kth.diva-portal.org/smash/get/diva2:1679022/FULLTEXT01.pdf

    lambda_dark = 0#promedio de fotones generados por bin
    dark_counts = np.random.poisson(lam=lambda_dark, size=numbins)
    det_input_dark = det_input + dark_counts
    det_output_dark = np.zeros(numbins//window)
    i=0
    for i in range(numbins//window): det_output_dark[i] = np.sum(det_input_dark[i*window:(i+1)*window])
    mean_detinput_dark = np.mean(det_input_dark)
    mean_detoutput_dark = np.mean(det_output_dark)/window
    var_detinput_dark = np.var(det_input_dark)
    var_detoutput_dark = np.var(det_output_dark)/window

    # print(f"Mean det_input_dark: {mean_detinput_dark}, Variance det_input_dark: {var_detinput_dark}")
    # print(f"Mean det_output_dark: {mean_detoutput_dark}, Variance det_output_dark: {var_detoutput_dark}")

    #la serie temporal del otro detector esta dada por 
    det_input2 = photons - det_input
    dark_counts2 = np.random.poisson(lam=lambda_dark, size=numbins)
    det2_input_dark = det_input2 + dark_counts2
    det2_output_dark = np.zeros(numbins//window)
    i=0
    for i in range(numbins//window): det2_output_dark[i] = np.sum(det2_input_dark[i*window:(i+1)*window])
    mean_det2input_dark = np.mean(det2_input_dark)
    mean_det2output_dark = np.mean(det2_output_dark)/window
    var_det2input_dark = np.var(det2_input_dark)
    var_det2output_dark = np.var(det2_output_dark)/window

    # print(f"Mean det2_input_dark: {mean_det2input_dark}, Variance det2_input_dark: {var_det2input_dark}")
    # print(f"Mean det2_output_dark: {mean_det2output_dark}, Variance det2_output_dark: {var_det2output_dark}")

    #calculo g2
    g2q = np.mean(det_output_dark*det2_output_dark)/(mean_detoutput_dark*mean_det2output_dark)/window**2
    # print(f"g2 = {g2}")

    #g2q = np.sum(det_output_dark*det2_output_dark)/(np.sum(det_output_dark)*np.sum(det2_output_dark))*numbins/window
    print(f"g2q = {g2q}")
    g2[j] = g2q

plt.plot(windows,g2,'r.--', label='Numerical Simulation without dark noise', markersize=10, linewidth=2)
x = np.linspace(1,100,100)
dark = 0
NSR = 2*dark
noise_factor = (1+NSR)**2

plt.plot(windows,1-(1/windows),'g--', label='Theoretical approximation', linewidth=2)

plt.legend(fontsize=18)
plt.grid()
plt.title('g2 vs. binning window for a source with g2(0)=0', fontsize=20) #xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None
plt.xlabel('Binning window (number of bins)', fontsize=18)
plt.ylabel('g2', fontsize=18)

plt.show()
