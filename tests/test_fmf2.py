#!/usr/bin/env python3

"""
Generic library tests, not dependent on external data
"""

import fmf2
import numpy as np
import pytest


@pytest.fixture
def zero_data():
    """Helper method to generate input data filled with zeros"""
    template = np.zeros((1, 1, 1, 1), dtype=np.float32)
    moveouts = np.zeros((1, 1, 1), dtype=np.int32)
    weights = np.zeros((1, 1, 1), dtype=np.float32)
    data = np.zeros((1, 1, 1), dtype=np.float32)
    return (template, moveouts, weights, data)


def test_missing_sycl(zero_data):
    """Check that if library isn't built with SYCL backend it raises an exception"""
    if 'sycl' in fmf2.AVAILABLE_BACKENDS:
        pytest.skip("SYCL backend exist, can't check for it being not present")
    (template, moveouts, weights, data) = zero_data
    with pytest.raises(RuntimeError) as err:
        fmf2.matched_filter(template, moveouts, weights, data, 1, arch='sycl')
    assert "not compiled with SYCL backend" in str(err.value)


def test_wrong_arch(zero_data):
    """Check that library raises exception with wrong or strange arch"""
    (template, moveouts, weights, data) = zero_data
    arch = 'test'
    with pytest.raises(NotImplementedError) as err:
        fmf2.matched_filter(template, moveouts, weights, data, 1, arch=arch)
    assert f'Unknown \'arch\': {arch}' in str(err.value)


def test_capitalized_arch(zero_data):
    """Check that wrong capitalization doesn't affect 'arch' parameter"""
    (template, moveouts, weights, data) = zero_data
    # The following calls must not raise the 
    fmf2.matched_filter(template, moveouts, weights, data, 1, arch='CPU')
    fmf2.matched_filter(template, moveouts, weights, data, 1, arch='CPu')
    fmf2.matched_filter(template, moveouts, weights, data, 1, arch='cPu')


def test_spaced_arch(zero_data):
    """Check that spaces doesn't affect 'arch' parameter"""
    (template, moveouts, weights, data) = zero_data
    # Add in 'accidental' spaces
    fmf2.matched_filter(template, moveouts, weights, data, 1, arch='cpu ')
    fmf2.matched_filter(template, moveouts, weights, data, 1, arch=' cpu ')
