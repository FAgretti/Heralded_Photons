#include "nlsesolver.h"
#include <math.h>
#include <fftw3.h>
#include <stdlib.h>

// Helper: compute |x|^2 for complex double
static inline double abs2(double complex x) {
    return creal(x)*creal(x) + cimag(x)*cimag(x);
}

// dBdz for NLSE: out = exp(-D*z) * FFT[ i*gamma*|A_t|^2*A_t ]
void dBdz_c(int N, double z, double complex* B, double complex* D, double gamma, double complex* out) {
    // Allocate arrays
    double complex* A_w = (double complex*) fftw_malloc(sizeof(double complex) * N);
    double complex* A_t = (double complex*) fftw_malloc(sizeof(double complex) * N);
    double complex* op_nolin = (double complex*) fftw_malloc(sizeof(double complex) * N);

    // A_w = B * exp(D*z)
    for (int i = 0; i < N; ++i) {
        A_w[i] = B[i] * cexp(D[i]*z);
    }

    // IFFT: A_t = IFFT(A_w)
    fftw_plan ifft_plan = fftw_plan_dft_1d(N, (fftw_complex*)A_w, (fftw_complex*)A_t, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(ifft_plan);
    // Normalize IFFT
    for (int i = 0; i < N; ++i) {
        A_t[i] /= N;
    }
    fftw_destroy_plan(ifft_plan);

    // op_nolin = FFT[ i*gamma*|A_t|^2*A_t ]
    for (int i = 0; i < N; ++i) {
        op_nolin[i] = I * gamma * abs2(A_t[i]) * A_t[i];
    }
    fftw_plan fft_plan = fftw_plan_dft_1d(N, (fftw_complex*)op_nolin, (fftw_complex*)op_nolin, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(fft_plan);
    fftw_destroy_plan(fft_plan);

    // out = exp(-D*z) * op_nolin
    for (int i = 0; i < N; ++i) {
        out[i] = cexp(-D[i]*z) * op_nolin[i];
    }

    fftw_free(A_w);
    fftw_free(A_t);
    fftw_free(op_nolin);
}
