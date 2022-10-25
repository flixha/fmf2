#!/usr/bin/env python3

"""
FMF2 is an efficient seismic matched-filter search library.

The main function of the library is the `matched_filter` method, available backends can be queried from the
`AVAILABLE_BACKENDS` list.
"""

from ._fmf2 import matched_filter, AVAILABLE_BACKENDS

__all__ = ('matched_filter', 'AVAILABLE_BACKENDS')
__author__ = ['Joergen Nordmoen <jorgen@nordmoen.net>', 'Felix Halpaap <felix.halpaap@uib.no>']
__version__ = '0.1.1'
