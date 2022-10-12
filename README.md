# FMF2 - seismic matched-filter search

`fmf2` is an efficient seismic matched-filter search, with inspiration from
[`fast_matched_filter`](https://github.com/beridel/fast_matched_filter). The
main difference is that FMF2 uses [SYCL](https://github.com/illuhad/hipSYCL) as
its GPU backend and a slightly updated build process based around
[Cython](https://cython.org/) and
[scikit-build](https://github.com/scikit-build/scikit-build).
