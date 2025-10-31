import numpy as np
import ctypes
import os

# Load the shared library (DLL/SO)
if os.name == 'nt':
    libname = 'nlsesolver.dll'
else:
    libname = 'libnlsesolver.so'

libpath = os.path.join(os.path.dirname(__file__), libname)
nlselib = ctypes.CDLL(r'C:\Users\Usuario\Documents\Balseiro-Ing Telecomunicaciones\PI\Simulaciones\nlse-pocosFotones\nlse_c\nlsesolver.dll')

# Define argument types for dBdz_c
nlselib.dBdz_c.argtypes = [
    ctypes.c_int, ctypes.c_double,
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
    ctypes.c_double,
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'),
]

def dBdz_c(N, z, B, D, gamma):
    out = np.zeros(N, dtype=np.complex128)
    nlselib.dBdz_c(N, z, B, D, gamma, out)
    return out
