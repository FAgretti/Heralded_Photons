#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Simulación de generación de pares y cálculo de CAR
// Parámetros barridos: tasa de pares, tasa de ruido, ventana de coincidencia, eficiencia

#define N_EVENTS 100000

int main() {
    // Barrido de parámetros
    double rates[] = {1e4, 5e4, 1e5, 2e5, 5e5}; // tasa de pares (Hz)
    double noises[] = {100, 500, 1000, 5000};    // tasa de dark counts (Hz)
    double windows[] = {1e-9, 2e-9, 5e-9, 1e-8, 2e-8, 5e-8, 1e-7}; // ventana (s)
    double eff = 0.2; // eficiencia de detección
    double T = 1.0;   // tiempo total de simulación (s)

    FILE *f = fopen("car_simulation.csv", "w");
    fprintf(f, "rate,noise,window,car\n");
    srand((unsigned)time(NULL));

    for (int ir=0; ir<sizeof(rates)/sizeof(rates[0]); ir++) {
        for (int inoise=0; inoise<sizeof(noises)/sizeof(noises[0]); inoise++) {
            for (int iw=0; iw<sizeof(windows)/sizeof(windows[0]); iw++) {
                double rate = rates[ir];
                double noise = noises[inoise];
                double window = windows[iw];
                int N = (int)(T * (rate + 2*noise));
                // Generar eventos para signal e idler
                double *signal = malloc(N * sizeof(double));
                double *idler  = malloc(N * sizeof(double));
                int n_signal=0, n_idler=0;
                // Pares verdaderos
                int n_pairs = (int)(T * rate);
                for (int i=0; i<n_pairs; i++) {
                    double t = ((double)rand()/RAND_MAX) * T;
                    if (((double)rand()/RAND_MAX) < eff) signal[n_signal++] = t;
                    if (((double)rand()/RAND_MAX) < eff) idler[n_idler++] = t;
                }
                // Ruido (dark counts)
                int n_noise = (int)(T * noise);
                for (int i=0; i<n_noise; i++) {
                    double t1 = ((double)rand()/RAND_MAX) * T;
                    double t2 = ((double)rand()/RAND_MAX) * T;
                    if (((double)rand()/RAND_MAX) < eff) signal[n_signal++] = t1;
                    if (((double)rand()/RAND_MAX) < eff) idler[n_idler++] = t2;
                }
                // Ordenar (opcional, pero útil para eficiencia)
                // Simple bubble sort (ok para N pequeño)
                for (int i=0; i<n_signal-1; i++) for (int j=0; j<n_signal-i-1; j++) if (signal[j]>signal[j+1]) { double tmp=signal[j]; signal[j]=signal[j+1]; signal[j+1]=tmp; }
                for (int i=0; i<n_idler-1; i++) for (int j=0; j<n_idler-i-1; j++) if (idler[j]>idler[j+1]) { double tmp=idler[j]; idler[j]=idler[j+1]; idler[j+1]=tmp; }
                // Contar coincidencias
                int coincidences=0, accidentals=0;
                int is=0, ii=0;
                while (is<n_signal && ii<n_idler) {
                    double dt = signal[is] - idler[ii];
                    if (fabs(dt) <= window/2) { coincidences++; is++; ii++; }
                    else if (dt < 0) is++;
                    else ii++;
                }
                // Accidentales: desplazar idler artificialmente
                int is2=0, ii2=0;
                double delay = window*10; // delay grande
                while (is2<n_signal && ii2<n_idler) {
                    double dt = signal[is2] - (idler[ii2]+delay);
                    if (fabs(dt) <= window/2) { accidentals++; is2++; ii2++; }
                    else if (dt < 0) is2++;
                    else ii2++;
                }
                double car = (accidentals>0) ? ((double)coincidences/accidentals) : 0.0;
                fprintf(f, "%g,%g,%g,%g\n", rate, noise, window, car);
                free(signal); free(idler);
            }
        }
    }
    fclose(f);
    return 0;
}
