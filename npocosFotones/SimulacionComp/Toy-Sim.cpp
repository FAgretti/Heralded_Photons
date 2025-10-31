// Toy-Sim.cpp
// C++ version of the photon correlation toy simulation sweep
// Results are saved to g2_sweep_results.csv for later analysis


#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif


using namespace std;

// Random number generator
std::mt19937 rng((unsigned int)std::chrono::steady_clock::now().time_since_epoch().count());

// Generate random double in [a, b)
double rand_uniform(double a, double b) {
    std::uniform_real_distribution<double> dist(a, b);
    return dist(rng);
}

// Generate random integer in [a, b]
int rand_int(int a, int b) {
    std::uniform_int_distribution<int> dist(a, b);
    return dist(rng);
}

// Generate photon arrival times for n_bins, each bin_duration
vector<double> generar_fotones(int n_bins, double bin_duration) {
    vector<double> tiempos;
    for (int i = 0; i < n_bins; ++i) {
        double t_base = i * bin_duration;
        for (int k = 0; k < 2; ++k) { // two photons per bin
            double jitter = rand_uniform(0, bin_duration);
            tiempos.push_back(t_base + jitter);
        }
    }
    return tiempos;
}

// Beam splitter and filter (simplified, just random split)
void split_y_filtrar(const vector<double>& fotones, vector<double>& t1, vector<double>& t2, vector<double>& t3) {
    vector<double> a, b;
    for (double t : fotones) {
        if (rand_uniform(0, 1) < 0.5) a.push_back(t);
        else b.push_back(t);
    }
    // Second beam splitter
    for (double t : b) {
        if (rand_uniform(0, 1) < 0.5) t2.push_back(t);
        else t3.push_back(t);
    }
    t1 = a;
}

// Add dark counts
void agregar_cuentas_oscuras(vector<double>& t, int n_dark, double duracion) {
    for (int i = 0; i < n_dark; ++i) {
        t.push_back(rand_uniform(0, duracion));
    }
    sort(t.begin(), t.end());
}

// Coincidences between two sorted vectors
template<typename T>
int coincidencias(const vector<T>& t1, const vector<T>& t2, double ventana) {
    int i = 0, j = 0, count = 0;
    while (i < t1.size() && j < t2.size()) {
        double dt = t1[i] - t2[j];
        if (abs(dt) <= ventana) {
            ++count; ++i; ++j;
        } else if (dt < -ventana) {
            ++i;
        } else {
            ++j;
        }
    }
    return count;
}

// Triple coincidences
int coincidencias_triples(const vector<double>& t1, const vector<double>& t2, const vector<double>& t3, double ventana) {
    int n123 = 0;
    for (double t : t1) {
        auto jt = lower_bound(t2.begin(), t2.end(), t - ventana);
        auto kt = lower_bound(t3.begin(), t3.end(), t - ventana);
        bool found = false;
        while (jt != t2.end() && *jt <= t + ventana && !found) {
            while (kt != t3.end() && *kt <= t + ventana && !found) {
                if (abs(*jt - *kt) <= ventana) {
                    ++n123; found = true;
                }
                ++kt;
            }
            ++jt;
        }
    }
    return n123;
}

// Detection efficiency
void detectar(double eff, vector<double>& t) {
    vector<double> out;
    for (double x : t) {
        if (rand_uniform(0, 1) < eff) out.push_back(x);
    }
    t = out;
}

// Main simulation
// Returns g2

double sim(int n_bins, double bin_duration, double ventana, int n_dark, double duracion, double eff) {
    auto fotones = generar_fotones(n_bins, bin_duration);
    vector<double> t1, t2, t3;
    split_y_filtrar(fotones, t1, t2, t3);
    detectar(eff, t1);
    detectar(eff, t2);
    detectar(eff, t3);
    agregar_cuentas_oscuras(t1, n_dark, duracion);
    agregar_cuentas_oscuras(t2, n_dark, duracion);
    agregar_cuentas_oscuras(t3, n_dark, duracion);
    sort(t1.begin(), t1.end());
    sort(t2.begin(), t2.end());
    sort(t3.begin(), t3.end());
    int n12 = coincidencias(t1, t2, ventana);
    int n13 = coincidencias(t1, t3, ventana);
    int n123 = coincidencias_triples(t1, t2, t3, ventana);
    int N1 = t1.size();
    double g2 = (n12 > 0 && n13 > 0) ? (N1 * n123) / double(n12 * n13) : 0.0;
    return g2;
}

int main() {
    // Sweep parameters
    vector<double> eficiencia, dark_count;
    for (int i = 0; i < 3; ++i) eficiencia.push_back(0.9 - i * 0.25); // 0.9 to 0.1
    for (int i = 0; i < 100; ++i) dark_count.push_back(i * 20.0); // 0 to 1990, 100 points
    int n_bins = 1000;
    double bin_duration = 1.0;
    double ventana = 0.1;
    double duracion = n_bins * bin_duration;
    int n_avg = 200;
    vector<vector<double>> g2(eficiencia.size(), vector<double>(dark_count.size(), 0.0));
    for (size_t i = 0; i < eficiencia.size(); ++i) {
        for (size_t j = 0; j < dark_count.size(); ++j) {
            double g2_sum = 0.0;
            for (int k = 0; k < n_avg; ++k) {
                g2_sum += sim(n_bins, bin_duration, ventana, int(dark_count[j]), duracion, eficiencia[i]);
            }
            g2[i][j] = g2_sum / n_avg;
            cout << "Eff=" << eficiencia[i] << ", Dark=" << dark_count[j] << ", g2=" << g2[i][j] << endl;
        }
    }

    // Save results to CSV in executable directory
#ifdef _WIN32
    char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    std::string fullpath(path);
    size_t pos = fullpath.find_last_of("\\/");
    std::string outdir = (pos == std::string::npos) ? "./" : fullpath.substr(0, pos+1);
#else
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    std::string fullpath(result, (count > 0) ? count : 0);
    size_t pos = fullpath.find_last_of("/");
    std::string outdir = (pos == std::string::npos) ? "./" : fullpath.substr(0, pos+1);
#endif
    std::string outpath = outdir + "g2_sweep_results.csv";
    ofstream fout(outpath);
    fout << "efficiency,dark_count,g2\n";
    for (size_t i = 0; i < eficiencia.size(); ++i) {
        for (size_t j = 0; j < dark_count.size(); ++j) {
            fout << eficiencia[i] << "," << dark_count[j] << "," << g2[i][j] << "\n";
        }
    }
    fout.close();
    cout << "Results saved to " << outpath << endl;
    return 0;
}
