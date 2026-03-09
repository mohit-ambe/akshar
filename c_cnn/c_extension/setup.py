from setuptools import setup, Extension

ext = Extension(
    name="cnn",                 # import name: import cnn
    sources=["cnn.c"],          # adjust path if needed
)

setup(
    name="cnn",
    version="0.0.0",
    ext_modules=[ext],
)