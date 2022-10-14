#!/usr/bin/env python3

"""
Medium data set to benchmark performance

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
def mohn_data():
    """Helper method to load data"""
    fil = pathlib.Path(__file__).parent / 'data' / 'mohn_cluster_tribe_20220425.npz'
    data = np.load(fil)
    return data


@pytest.mark.slow
@pytest.mark.benchmark(group='mohn_cluster')
def test_cpu(mohn_data, benchmark):
    templates = mohn_data['templates']
    weights = mohn_data['weights']
    moveouts = mohn_data['moveouts']
    data = mohn_data['data']
    normalize = 'full'
    benchmark(fmf2.matched_filter, templates, moveouts, weights, data, step=1,
              arch='precise', normalize=normalize)
