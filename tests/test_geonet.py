#!/usr/bin/env python3

"""
Small test to ensure correct operations of the backend implementations

The example data was provided by Felix Halpaap

The data consists of the necessary numpy arrays needed to run through the
`matched_filter` method. In addition there are two arrays for comparison, one
created by the `fast_matched_filter`'s `precise` implementation (stored as
`cc_sums`) and one array produced by the EQcorrscan library (stored as
`cc_ref`)
"""

import fmf2
import numpy as np
import pathlib
import pytest


@pytest.fixture
def geonet_data():
    """Helper method to load data"""
    fil = pathlib.Path(__file__).parent / "data/geonet.npz"
    data = np.load(fil)
    return data


@pytest.mark.parametrize('impl', ['precise', 'sycl'])
def test_geonet(geonet_data, impl):
    if impl not in fmf2.AVAILABLE_BACKENDS:
        pytest.skip("Not compiled with backed: %s, available: %s" % (impl, fmf2.AVAILABLE_BACKENDS))
    templates = geonet_data['templates']
    weights = geonet_data['weights']
    moveouts = geonet_data['moveouts']
    data = geonet_data['data']
    normalize = 'full'
    result = fmf2.matched_filter(templates, moveouts, weights, data, step=1, arch=impl, normalize=normalize)
    assert np.allclose(result, geonet_data['cc_sums'], atol=1e-5)
    assert np.allclose(result, geonet_data['cc_ref'], atol=1e-5)


@pytest.mark.parametrize('check_zeros', [None, 'first', 'all'])
def test_check_zeros(geonet_data, capsys, check_zeros):
    templates = geonet_data['templates']
    weights = geonet_data['weights']
    moveouts = geonet_data['moveouts']
    data = geonet_data['data']
    normalize = 'full'
    result = fmf2.matched_filter(templates, moveouts, weights, data, step=1, arch='cpu', normalize=normalize,
                                 check_zeros=check_zeros)
    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err
    assert np.allclose(result, geonet_data['cc_sums'], atol=1e-5)
    assert np.allclose(result, geonet_data['cc_ref'], atol=1e-5)


@pytest.mark.benchmark(group="geonet")
@pytest.mark.parametrize('impl', ['precise', 'sycl'])
def test_bench(geonet_data, impl, benchmark):
    if impl not in fmf2.AVAILABLE_BACKENDS:
        pytest.skip("Not compiled with backed: %s, available: %s" % (impl, fmf2.AVAILABLE_BACKENDS))
    templates = geonet_data['templates']
    weights = geonet_data['weights']
    moveouts = geonet_data['moveouts']
    data = geonet_data['data']
    normalize = 'full'
    benchmark(fmf2.matched_filter, templates, moveouts, weights, data, step=1,
              arch=impl, normalize=normalize)
