# NLSE C Solver

This folder contains a fast C implementation of the NLSE right-hand side for use in Python.

## Files
- `nlsesolver.h` / `nlsesolver.c`: C code for the NLSE ODE right-hand side using FFTW.
- `setup.py`: Build script for Python extension (requires FFTW and NumPy).
- `nlsesolver_python.py`: Python ctypes wrapper for the C code.

## Build Instructions

1. Install FFTW and NumPy (e.g., `conda install fftw numpy` or `pip install numpy` and install FFTW for your OS).
2. Build the shared library:
   - On Linux/Mac: `gcc -O3 -fPIC -shared nlsesolver.c -o libnlsesolver.so -lfftw3 -lm`
   - On Windows (MinGW): `gcc -O3 -shared -o nlsesolver.dll nlsesolver.c -lfftw3-3`
3. Use `nlsesolver_python.py` to call the C function from Python.

## Usage

See `nlsesolver_python.py` for an example of calling the C function from Python.

---

**Note:** You may need to adjust the build command and library names depending on your system and FFTW installation.
