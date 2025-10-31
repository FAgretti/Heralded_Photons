#ifndef NLSESOLVER_H
#define NLSESOLVER_H

#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

void dBdz_c(int N, double z, double complex* B, double complex* D, double gamma, double complex* out);

#ifdef __cplusplus
}
#endif

#endif // NLSESOLVER_H
