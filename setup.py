from skbuild import setup


setup(
    name="fmf2",
    version="0.1.0",
    description="Fast Matched Filter implementation with SYCL backend",
    author='jorgen@nordmoen.net',
    license="MIT",
    packages=['fmf2'],
    python_requires=">=3.7",
    install_requires=['numpy'],
    extras_require={
        'tests': ['pytest', 'pytest-benchmark'],
    },

)
