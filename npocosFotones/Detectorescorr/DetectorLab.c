// hbt.c — barrido de g2_heralded(0) con HBT, dark por ventana
// gcc -O3 -lm -o hbt hbt.c

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

static inline double urand() {
    // rand() / (RAND_MAX+1.0) evita exactamente 1.0
    return rand() / (RAND_MAX + 1.0);
}

static inline double rand_exponential(double lambda) {
    // inter-arrival para proceso de Poisson
    double u = urand();
    return -log(1.0 - u) / lambda;
}

// Genera tiempos de pares de un proceso Poisson de tasa "rate" en [0, total_time).
// Devuelve cantidad de pares y escribe el puntero a arreglo con tiempos (malloc).
static int generate_photon_pairs(double rate, double total_time, double **times_out) {
    // cota superior (promedio*1.5 + margen) para reservar; si queda corto, re-alloc
    int cap = (int)(rate * total_time * 1.5) + 1024;
    if (cap < 1024) cap = 1024;
    double *times = (double*)malloc((size_t)cap * sizeof(double));
    if (!times) { fprintf(stderr, "malloc times\n"); exit(1); }

    int n = 0;
    double t = 0.0;
    while (t < total_time) {
        double dt = rand_exponential(rate);
        t += dt;
        if (t >= total_time) break;
        if (n >= cap) {
            cap = (int)(cap * 1.6) + 1024;
            times = (double*)realloc(times, (size_t)cap * sizeof(double));
            if (!times) { fprintf(stderr, "realloc times\n"); exit(1); }
        }
        times[n++] = t;
    }
    *times_out = times;
    return n;
}

// Detectores no PNR: flags por ventana (click/no-click)
typedef struct {
    uint8_t H;   // click herald
    uint8_t S2;  // click detector 2
    uint8_t S3;  // click detector 3
} BinFlags;

// Simulación HBT heralded con dark por ventana y splitter 50/50 en signal.
// Devuelve contadores N1,N12,N13,N123 vía punteros.
static void simulate_hbt_heralded(
    int n_windows,
    double window_duration,
    double rate_pairs,
    double eta_herald, double eta_signal,
    double Rdark_H, double Rdark_2, double Rdark_3,
    unsigned long long *N1,
    unsigned long long *N12,
    unsigned long long *N13,
    unsigned long long *N123
){
    double total_time = (double)n_windows * window_duration;

    // 1) Generar pares
    double *pair_times = NULL;
    int n_pairs = generate_photon_pairs(rate_pairs, total_time, &pair_times);

    // 2) Flags por ventana
    BinFlags *bins = (BinFlags*)calloc((size_t)n_windows, sizeof(BinFlags));
    if (!bins) { fprintf(stderr, "calloc bins\n"); exit(1); }

    // 3) Contribución de pares reales (aplican eficiencias por fotón)
    for (int i = 0; i < n_pairs; ++i) {
        int k = (int)floor(pair_times[i] / window_duration);
        if (k < 0 || k >= n_windows) continue;

        // Herald detectado?
        if (urand() < eta_herald) {
            bins[k].H = 1;
        }

        // Signal detectado y a qué rama del splitter llega el click
        if (urand() < eta_signal) {
            if (urand() < 0.5) bins[k].S2 = 1;
            else               bins[k].S3 = 1;
        }
    }

    // 4) Oscuras por ventana (proceso independiente): p = 1 - exp(-Rdark * Δt)
    double pH = 1.0 - exp(-Rdark_H * window_duration);
    double p2 = 1.0 - exp(-Rdark_2 * window_duration);
    double p3 = 1.0 - exp(-Rdark_3 * window_duration);

    for (int k = 0; k < n_windows; ++k) {
        if (!bins[k].H  && urand() < pH) bins[k].H  = 1;
        if (!bins[k].S2 && urand() < p2) bins[k].S2 = 1;
        if (!bins[k].S3 && urand() < p3) bins[k].S3 = 1;
    }

    // 5) Contadores HBT heralded
    unsigned long long n1 = 0, n12 = 0, n13 = 0, n123 = 0;
    for (int k = 0; k < n_windows; ++k) {
        if (bins[k].H) {
            n1++;
            if (bins[k].S2) n12++;
            if (bins[k].S3) n13++;
            if (bins[k].S2 && bins[k].S3) n123++;
        }
    }

    // devolver
    *N1 = n1; *N12 = n12; *N13 = n13; *N123 = n123;

    free(pair_times);
    free(bins);
}

static inline double g2_heralded_from_counts(
    unsigned long long N1, unsigned long long N12,
    unsigned long long N13, unsigned long long N123
){
    if (N1 == 0 || N12 == 0 || N13 == 0) return NAN;
    return (double)N123 * (double)N1 / ((double)N12 * (double)N13);
}

int main(void){
    srand((unsigned int)time(NULL));

    // ======= Parámetros del barrido =======
    // Ventanas de integración (s)
    const int n_window = 2;
    const double window_list[2] = {1e-10, 2e-10};

    // Tasas de generación de pares (Hz)
    const int n_rate = 12;
    const double rate_list[12] = {1e8, 2e8, 5e8, 1e9, 2e9, 5e9, 1e10, 2e10, 5e10, 1e11, 2e11, 5e11, 1e12, 2e12};

    // Tasas de oscuras por detector (Hz) — podés ajustar o ampliar
    const int n_dark = 1;
    const double Rdark_list[1] = {0.5e10};

    // Eficiencias
    const double eta_H = 0.5;    // herald
    const double eta_S = 0.5;    // signal (antes del splitter)

    // Número de ventanas para promediar estadísticas
    // (subí si querés más precisión, baja si el runtime es largo)
    const int N_WINDOWS = 200000;  // 2e5

    // ======= Salida =======
    FILE *f = fopen("g2_heralded_hbt.csv", "w");
    if (!f){ fprintf(stderr, "No pude abrir salida\n"); return 1; }
    fprintf(f, "rate,window_duration,Rdark_H,Rdark_2,Rdark_3,eta_H,eta_S,N1,N12,N13,N123,g2_heralded\n");

    // Barrido
    for (int iw = 0; iw < n_window; ++iw) {
        double win = window_list[iw];
        for (int id = 0; id < n_dark; ++id) {
            double Rd = Rdark_list[id]; // usar la misma tasa en H, 2 y 3 (puede diferenciarse si querés)
            for (int ir = 0; ir < n_rate; ++ir) {
                double rate = rate_list[ir];

                unsigned long long N1=0, N12=0, N13=0, N123=0;
                simulate_hbt_heralded(
                    N_WINDOWS, win, rate, eta_H, eta_S,
                    Rd, Rd, Rd,
                    &N1, &N12, &N13, &N123
                );
                double g2 = g2_heralded_from_counts(N1, N12, N13, N123);

                fprintf(
                    f,
                    "%.6e,%.6e,%.6e,%.6e,%.6e,%.3f,%.3f,%llu,%llu,%llu,%llu,%.8f\n",
                    rate, win, Rd, Rd, Rd, eta_H, eta_S, N1, N12, N13, N123, g2
                );
            }
            fflush(f);
        }
    }
    fclose(f);

    fprintf(stdout, "Listo: g2_heralded_hbt.csv generado.\n");
    fprintf(stdout, "Columnas: rate, window_duration, Rdark_*, eta_*, N1,N12,N13,N123, g2_heralded\n");
    return 0;
}
