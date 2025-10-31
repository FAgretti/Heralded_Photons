#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Genera un número aleatorio uniforme entre 0 y 1
double rand_uniform() {
    return rand() / (RAND_MAX + 1.0);
}

// Genera un número aleatorio exponencial con media 1/lambda
double rand_exponential(double lambda) {
    double u = rand_uniform();
    return -log(1-u) / lambda;
}

// Simula la llegada de fotones antibunching
int generate_photon_arrivals(double rate, double total_time, double min_gap, double **times_out) {
    int max_photons = (int)(rate * total_time * 2); // sobreestimación
    double *times = malloc(max_photons * sizeof(double));
    int n = 0;
    double t = 0;
    while (t < total_time) {
        double dt = rand_exponential(rate);
        if (dt < min_gap) dt = min_gap;
        t += dt;
        if (t < total_time) {
            times[n++] = t;
        }
    }
    *times_out = times;
    return n;
}

double simulate_g2(int n_windows, double window_duration, double rate, double min_gap) {
    double total_time = n_windows * window_duration;
    double *photon_times;
    int n_photons = generate_photon_arrivals(rate, total_time, min_gap, &photon_times);

    // Beam splitter
    int *I1 = calloc(n_windows, sizeof(int));
    int *I2 = calloc(n_windows, sizeof(int));
    for (int i = 0; i < n_photons; ++i) {
        int det = rand_uniform() < 0.5 ? 1 : 2;
        int bin = (int)(photon_times[i] / window_duration);
        if (bin >= n_windows) continue;
        if (det == 1) I1[bin]++;
        else I2[bin]++;
    }

    // Calcular g2
    double sum_I1 = 0, sum_I2 = 0, sum_I1I2 = 0;
    for (int i = 0; i < n_windows; ++i) {
        sum_I1 += I1[i];
        sum_I2 += I2[i];
        sum_I1I2 += I1[i] * I2[i];
    }
    double mean_I1 = sum_I1 / n_windows;
    double mean_I2 = sum_I2 / n_windows;
    double mean_I1I2 = sum_I1I2 / n_windows;
    double g2 = (mean_I1 * mean_I2 > 0) ? mean_I1I2 / (mean_I1 * mean_I2) : 0;

    // Guardar intensidades solo si min_gap y window_duration son los primeros del barrido
    static int saved = 0;
    if (!saved) {
        FILE *fi = fopen("intensities.csv", "w");
        if (fi) {
            fprintf(fi, "I1,I2\n");
            for (int i = 0; i < n_windows; ++i) {
                fprintf(fi, "%d,%d\n", I1[i], I2[i]);
            }
            fclose(fi);
        }
        saved = 1;
    }

    free(photon_times);
    free(I1);
    free(I2);
    return g2;
}

int main() {
    srand((unsigned int)time(NULL));

    // === CONFIGURACIÓN DE VARIABLES BARRIDAS ===
    // Modifique aquí los parámetros del barrido:
    #define N_WINDOWS 100000
    #define PHOTON_RATE 1e9

    // min_gap sweep: log spacing with 1, 2, 5 per decade
    const int n_min_gap = 14;
    double min_gap_list[14] = {
        5e-10, 1e-9, 2e-9, 5e-9,
        1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6
        , 2e-6, 5e-6, 1e-5
    };

    // window_duration sweep: log spacing with 1, 2, 5 per decade
    const int n_window = 14;
    double window_list[14] = {
        5e-10, 1e-9, 2e-9, 5e-9,
        1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6
        , 2e-6, 5e-6, 1e-5
    };
    // ===========================================

    int n_windows = N_WINDOWS;
    double rate = PHOTON_RATE;

    // Abrir archivo para guardar resultados
    FILE *f = fopen("g2_barrido.csv", "w");
    fprintf(f, "min_gap,window_duration,g2\n");

    for (int i = 0; i < n_min_gap; ++i) {
        for (int j = 0; j < n_window; ++j) {
            double g2 = simulate_g2(n_windows, window_list[j], rate, min_gap_list[i]);
            printf("min_gap=%.1e, window=%.1e, g2=%.3f\n", min_gap_list[i], window_list[j], g2);
            fprintf(f, "%.2e,%.2e,%.6f\n", min_gap_list[i], window_list[j], g2);
        }
    }
    fclose(f);

    printf("Resultados guardados en g2_barrido.csv\n");
    return 0;
}