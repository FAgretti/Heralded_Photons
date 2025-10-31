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

// Simula la llegada de pares de fotones (emisión espontánea de pares)
int generate_photon_pairs(double rate, double total_time, double **times_out) {
    int max_pairs = (int)(rate * total_time * 2); // sobreestimación
    double *times = malloc(max_pairs * sizeof(double));
    int n = 0;
    double t = 0;
    while (t < total_time) {
        double dt = rand_exponential(rate);
        t += dt;
        if (t < total_time) {
            times[n++] = t;
        }
    }
    *times_out = times;
    return n;
}

double simulate_g2(int n_windows, double window_duration, double rate) {
    double total_time = n_windows * window_duration;
    double *photon_times;
    int n_photons = generate_photon_pairs(rate, total_time, &photon_times);

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

    // Guardar intensidades solo para el primer barrido
    static int saved = 0;
    if (!saved) {
        FILE *fi = fopen("intensities_pairs.csv", "w");
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
    #define N_WINDOWS 100000
    // Barrido sobre rate y window_duration (matriz g2)
    const int n_rate = 16;
    double rate_list[16] = {
        1e4, 2e4, 5e4,
        1e5, 2e5, 5e5,
        1e6, 2e6, 5e6,
        1e7, 2e7, 5e7,
        1e8, 2e8, 5e8,
        1e9
    };
    const int n_window = 12;
    double window_list[12] = {
        1e-10, 2e-10, 5e-10,
        1e-9, 2e-9, 5e-9,
        1e-8, 2e-8, 5e-8,
        1e-7, 2e-7, 5e-7
    };
    int n_windows = N_WINDOWS;

    FILE *f = fopen("g2_barrido_pairs.csv", "w");
    fprintf(f, "rate,window_duration,g2\n");

    for (int i = 0; i < n_rate; ++i) {
        double rate = rate_list[i];
        for (int j = 0; j < n_window; ++j) {
            double window_duration = window_list[j];
            double g2 = simulate_g2(n_windows, window_duration, rate);
            printf("rate=%.1e, window=%.1e, g2=%.3f\n", rate, window_duration, g2);
            fprintf(f, "%.2e,%.2e,%.6f\n", rate, window_duration, g2);
        }
    }
    fclose(f);

    printf("Resultados guardados en g2_barrido_pairs.csv\n");
    return 0;
}
