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

// Simula la llegada de pares de fotones (heralded)
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

// Simula detección con eficiencia
int detect(double efficiency) {
    return rand_uniform() < efficiency;
}

// Simula g2 heralded (condicionado a detección herald)
double simulate_heralded_g2(int n_windows, double window_duration, double rate, double eff_herald, double eff_signal) {
    double total_time = n_windows * window_duration;
    double *pair_times;
    int n_pairs = generate_photon_pairs(rate, total_time, &pair_times);

    // Arrays para clic/no clic en ventanas heralded
    int *heralded_click = calloc(n_windows, sizeof(int)); // 1 si hubo clic en signal en ventana heralded
    int *I_herald = calloc(n_windows, sizeof(int));

    for (int i = 0; i < n_pairs; ++i) {
        int bin = (int)(pair_times[i] / window_duration);
        if (bin >= n_windows) continue;
        int detected_herald = detect(eff_herald);
        int detected_signal = detect(eff_signal);
        if (detected_herald) {
            I_herald[bin]++;
            if (detected_signal) {
                heralded_click[bin] = 1; // basta con un clic
            }
        }
    }

    // Calcular fracción de ventanas heralded con clic
    int n_heralded = 0, n_heralded_click = 0;
    for (int i = 0; i < n_windows; ++i) {
        if (I_herald[i] > 0) {
            n_heralded++;
            if (heralded_click[i]) n_heralded_click++;
        }
    }
    double p_click_heralded = (n_heralded > 0) ? ((double)n_heralded_click / n_heralded) : 0;

    // Guardar intensidades solo para el primer barrido
    static int saved = 0;
    if (!saved) {
        FILE *fi = fopen("intensities_heralded.csv", "w");
        if (fi) {
            fprintf(fi, "heralded_click,I_herald\n");
            for (int i = 0; i < n_windows; ++i) {
                fprintf(fi, "%d,%d\n", heralded_click[i], I_herald[i]);
            }
            fclose(fi);
        }
        saved = 1;
    }

    free(pair_times);
    free(heralded_click);
    free(I_herald);
    return p_click_heralded;
}

int main() {
    srand((unsigned int)time(NULL));

    // === CONFIGURACIÓN DE VARIABLES BARRIDAS ===
    #define N_WINDOWS 100000
    // Barrido sobre rate y window_duration (matriz g2)
    const int n_rate = 10;
    double rate_list[10] = {
        1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9
    };
    const int n_window = 8;
    double window_list[8] = {
        1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6
    };
    int n_windows = N_WINDOWS;
    double eff_herald = 0.5;  // Eficiencia de detección herald
    double eff_signal = 0.5;  // Eficiencia de detección signal

    FILE *f = fopen("g2_barrido_heralded.csv", "w");
    fprintf(f, "rate,window_duration,g2_heralded\n");

    for (int i = 0; i < n_rate; ++i) {
        double rate = rate_list[i];
        for (int j = 0; j < n_window; ++j) {
            double window_duration = window_list[j];
            double g2 = simulate_heralded_g2(n_windows, window_duration, rate, eff_herald, eff_signal);
            printf("rate=%.1e, window=%.1e, g2_heralded=%.3f\n", rate, window_duration, g2);
            fprintf(f, "%.2e,%.2e,%.6f\n", rate, window_duration, g2);
        }
    }
    fclose(f);

    printf("Resultados guardados en g2_barrido_heralded.csv\n");
    return 0;
}
