#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Genera un número aleatorio uniforme entre 0 y 1
double rand_uniform() {
    return rand() / (RAND_MAX + 1.0);
}

double rand_exponential(double lambda) {
    double u = rand_uniform();
    return -log(1-u) / lambda;
}

// Simula la llegada de pares de fotones (heralded)
int generate_photon_pairs(double rate, double total_time, double **times_out) {
    int max_pairs = (int)(rate * total_time * 2);
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

int detect(double efficiency) {
    return rand_uniform() < efficiency;
}

int dark_count(double p_dark) {
    return rand_uniform() < p_dark;
}

// Simulación 1: Probabilidad de clic heralded con ruido
// Devuelve la fracción de ventanas heralded con clic (incluyendo ruido)
double simulate_heralded_with_noise(int n_windows, double window_duration, double rate, double eff_herald, double eff_signal, double p_dark) {
    double total_time = n_windows * window_duration;
    double *pair_times;
    int n_pairs = generate_photon_pairs(rate, total_time, &pair_times);
    int *heralded_click = calloc(n_windows, sizeof(int));
    int *I_herald = calloc(n_windows, sizeof(int));

    for (int i = 0; i < n_pairs; ++i) {
        int bin = (int)(pair_times[i] / window_duration);
        if (bin >= n_windows) continue;
        int detected_herald = detect(eff_herald);
        int detected_signal = detect(eff_signal);
        int noise = dark_count(p_dark);
        if (detected_herald) {
            I_herald[bin]++;
            if (detected_signal || noise) {
                heralded_click[bin] = 1;
            }
        }
    }
    int n_heralded = 0, n_heralded_click = 0;
    for (int i = 0; i < n_windows; ++i) {
        if (I_herald[i] > 0) {
            n_heralded++;
            if (heralded_click[i]) n_heralded_click++;
        }
    }
    double p_click_heralded = (n_heralded > 0) ? ((double)n_heralded_click / n_heralded) : 0;
    free(pair_times); free(heralded_click); free(I_herald);
    return p_click_heralded;
}

// Simulación 2: Coincidencia cruzada entre detectores (con ruido y eficiencia)

// (El código de apertura de archivo y el bucle deben estar dentro de main)

double simulate_cross_correlation(int n_windows, double window_duration, double rate, double eff1, double eff2, double p_dark1, double p_dark2) {
    double total_time = n_windows * window_duration;
    double *pair_times;
    int n_pairs = generate_photon_pairs(rate, total_time, &pair_times);
    int *D1 = calloc(n_windows, sizeof(int));
    int *D2 = calloc(n_windows, sizeof(int));
    for (int i = 0; i < n_pairs; ++i) {
        int bin = (int)(pair_times[i] / window_duration);
        if (bin >= n_windows) continue;
        if (detect(eff1) || dark_count(p_dark1)) D1[bin]++;
        if (detect(eff2) || dark_count(p_dark2)) D2[bin]++;
    }
    // Calcular coincidencias cruzadas
    double sum_D1 = 0, sum_D2 = 0, sum_D1D2 = 0;
    for (int i = 0; i < n_windows; ++i) {
        sum_D1 += D1[i] > 0 ? 1 : 0;
        sum_D2 += D2[i] > 0 ? 1 : 0;
        sum_D1D2 += (D1[i] > 0 && D2[i] > 0) ? 1 : 0;
    }
    double mean_D1 = sum_D1 / n_windows;
    double mean_D2 = sum_D2 / n_windows;
    double mean_D1D2 = sum_D1D2 / n_windows;
    double g2_cross = (mean_D1 * mean_D2 > 0) ? mean_D1D2 / (mean_D1 * mean_D2) : 0;
    free(pair_times); free(D1); free(D2);
    return g2_cross;
}

int main() {
    srand((unsigned int)time(NULL));
    #define N_WINDOWS 100000
    const int n_rate = 9;
    double rate_list[9] = {1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8};
    const int n_window = 7;
    double window_list[7] = {1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6};
    double dark_p[8] = {1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    int n_windows = N_WINDOWS;
    double eff1 = 0.5, eff2 = 0.5;
    //double p_dark1 = 1e-4, p_dark2 = 1e-4; // Probabilidad de dark count por ventana

    // Simulación 1: clic heralded con ruido
    FILE *f1 = fopen("heralded_with_noise.csv", "w");
    fprintf(f1, "rate,window_duration,p_click_heralded\n");
    for (int i = 0; i < n_rate; ++i) {
        double rate = rate_list[i];
        for (int j = 0; j < n_window; ++j) {
            for(int k = 0; k < 8; ++k) {
                double p_dark2 = dark_p[k];
                double window_duration = window_list[j];
                double p_click = simulate_heralded_with_noise(n_windows, window_duration, rate, eff1, eff2, p_dark2);
                printf("rate=%.1e, window=%.1e, p_dark2=%.1e, p_click_heralded=%.3f\n", rate, window_duration, p_dark2, p_click);
                fprintf(f1, "%.2e,%.2e,%.1e,%.6f\n", rate, window_duration, p_dark2, p_click);
            }
        }
    }
    fclose(f1);

    // Simulación 2: coincidencia cruzada con ruido y eficiencia
    FILE *f2 = fopen("cross_correlation.csv", "w");
    fprintf(f2, "rate,window_duration,g2_cross\n");
    for (int i = 0; i < n_rate; ++i) {
        double rate = rate_list[i];
        for (int j = 0; j < n_window; ++j) {
            for(int k = 0; k < 8; ++k) {
                double p_dark2 = dark_p[k];
                double p_dark1 = dark_p[k];
                double window_duration = window_list[j];
                double g2_cross = simulate_cross_correlation(n_windows, window_duration, rate, eff1, eff2, p_dark1, p_dark2);
                printf("rate=%.1e, window=%.1e, g2_cross=%.3f\n", rate, window_duration, g2_cross);
                fprintf(f2, "%.2e,%.2e,%.6f\n", rate, window_duration, g2_cross);
            }
        }
    }
    fclose(f2);

    printf("Resultados guardados en heralded_with_noise.csv y cross_correlation.csv\n");
    return 0;
}
