from setuptools import setup, Extension
import numpy

nlsesolver_module = Extension(
    'nlsesolver',
    sources=['nlsesolver.c'],
    include_dirs=[numpy.get_include()],
    libraries=['fftw3'],
)

setup(
    name='nlsesolver',
    version='0.1',
    ext_modules=[nlsesolver_module],
)
