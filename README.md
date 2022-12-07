# FMF2 - seismic matched-filter search
[![Build library](https://github.com/nordmoen/fmf2/actions/workflows/python-package.yml/badge.svg)](https://github.com/nordmoen/fmf2/actions/workflows/python-package.yml)

`fmf2` is an efficient seismic matched-filter search, with inspiration from
[`fast_matched_filter`](https://github.com/beridel/fast_matched_filter). The
main difference is that FMF2 uses [SYCL](https://github.com/illuhad/hipSYCL) as
its GPU backend and a slightly updated build process based around
[Cython](https://cython.org/) and
[scikit-build](https://github.com/scikit-build/scikit-build).

## Building

To build this library one needs a recent Python version, a C compiler and
CMake. Building is taken care of by Python, which eventually will call out to
CMake. It is recommended to create a virtual environment before installing the
library. For the GPU backend hipSYCL is required, but this will be dynamically
detected at compile time and enabled/disabled based on availability.

```bash
# Optional:
python3 -m venv fmf2_env
source fmf2_env/bin/activate

# Build and install the library
python3 -m pip -v install .
```

If you are rebuilding this library and it fails without an apparent cause, a
trick can be to remove the temporary `_skbuild/` directory.

### Build options

To change the build options when building with Python prepend the build command
with `CMAKE_ARGS="-DOPTION=..."` to inform CMake.

```bash
CMAKE_ARGS="-DCPU_SKIP=OFF" python3 -m pip -v install .
```

When the weights matrix is very sparse (i.e., many weights set to zero, e.g., due
to very heterogeneous station configurations for each template), then the total
runtime may be reduced by skipping correlations for zero-weights even on the GPU.
But skipping traces may degrade performance due to worse cache locality, so you
should test the performance with and without skipping.
```bash
CMAKE_ARGS="-DGPU_SKIP=ON" python3 -m pip -v install .
```

### Testing

To test the installation one can use `pytest`:

```bash
# Install test dependencies
python3 -m pip install pytest pytest-benchmark

# Run tests
pytest -m "not slow"
```

The above test command will run the quick verification tests that finish within
a few seconds. Removing the `-m "not slow"` will run the longer performance
benchmark which can be used to ensure good performance of the different
backends. Note that the performance benchmark requires additional datasets that
are too big to include in git.
