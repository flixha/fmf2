import numpy as np
cimport numpy as np

cdef extern from "fmf2.hpp":
    int matched_filter_serial(const float* templates, const float* sum_square_templates, const int* moveouts, const float* data,
                  const float* weights, const int step, const int n_samples_template,
                  const int n_samples_data, const int n_templates,
                  const int n_stations, const int n_components, const int n_corr,
                  const int normalize, float* cc_sum);

    int matched_filter_sycl(const float* templates, const float* sum_square_templates, const int* moveouts, const float* data,
                const float* weights, const int step, const int n_samples_template,
                const int n_samples_data, const int n_templates,
                const int n_stations, const int n_components, const int n_corr,
                const int normalize, float* cc_sum);

# The input field must have the following requirements so that we can safely
# pass it along to C/C++ code without danger of reading unintended memory
cdef list NP_FIELD_REQUIREMENTS = ['C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE']

# List of available backends
AVAILABLE_BACKENDS = ['cpu', 'precise']
IF FMF2_SYCL:
    AVAILABLE_BACKENDS.append('sycl')
    AVAILABLE_BACKENDS.append('gpu')

def matched_filter(templates, moveouts, weights, data, step, arch='cpu',
        check_zeros=None, normalize='short'):
    """Compute the correlation coefficients between `templates` and `data`.

    Scan the continuous waveforms `data` with the template waveforms
    `templates` given the relative propagation times `moveouts` and compute
    a time series of summed correlation coefficients. The weighted sum is
    defined by `weights`. Try `normalize='full'` and/or `arch='precise' or 'gpu'`
    to achieve better numerical precision.

    Parameters
    -----------
    templates: numpy.ndarray
        4D (n_templates, n_stations, n_channels, n_tp_samples) or 3D 
        (n_templates, n_traces, n_tp_samples) `numpy.ndarray` with the
        template waveforms.
    moveouts: numpy.ndarray, int
        3D (n_templates, n_stations, n_channels) or 2D (n_templates, n_stations)
        `numpy.ndarray` with the moveouts, in samples.
    weights: numpy.ndarray, float
        3D (n_templates, n_stations, n_channels) or 2D (n_stations, n_channels)
        `numpy.ndarray` with the channel weights. For a given template, the
        largest possible correlation coefficient is given by the sum of the
        weights. Make sure that the weights sum to one if you want CCs between
        1 and -1.
    data: numpy.ndarray
        3D (n_stations, n_channels, n_samples) or 2D (n_traces, n_samples)
        `numpy.ndarray` with the continuous waveforms.
    step: scalar, int
        Time interval, in samples, between consecutive correlations.
    arch: string, optional
        One `'cpu'`, `'precise'` or `'gpu'`. The `'precise'` implementation
        is a CPU implementation that slower but more accurate than `'cpu'`.
        The GPU implementation is used if `arch='gpu'`. Default is `'cpu'`.
    check_zeros: string, optional
        Controls the verbosity level at the end of this routine when
        checking zeros in the time series of correlation coefficients (CCs).
        - False: No messages.  
        - `'first'`: Check zeros on the first template's CCs (recommended).  
        - `'all'`: Check zeros on each template's CCs. It can be useful for
        troubleshooting but in general this would print too many messages.  
        Default is `'first'`.
    normalize: string, optional
        Either "short" or "full" - full is slower but removes the mean of the
        data at every correlation. Short is the original implementation.
        NB: When using normalize="short", the templates and the data sliding
        windows must have zero means (high-pass filter the data if necessary).

    Returns
    --------
    cc_sums: numpy.ndarray, float
        2D (n_templates, n_correlations) `numpy.ndarray`. The number of
        correlations is controlled by `step`.
    """
    norm = 1 if normalize.strip().lower() == 'full' else 0

    # figure out and check input formats
    impossible_dimensions = False
    if templates.ndim > data.ndim:
        n_templates = int(templates.shape[0])

        assert templates.shape[1] == data.shape[0] # check stations
        n_stations = int(templates.shape[1])

        if templates.ndim == 4:
            assert templates.shape[2] == data.shape[1] # check components
            n_components = int(templates.shape[2])
        elif templates.ndim == 3:
            n_components = int(1)
        else:
            impossible_dimensions = True

    elif templates.ndim == data.ndim:
        n_templates = int(1)
        
        assert templates.shape[0] == data.shape[0] # check stations
        n_stations = int(templates.shape[0])

        if templates.ndim == 3:
            assert templates.shape[1] == data.shape[1] # check components
            n_components = int(templates.shape[1])
        elif templates.ndim == 2:
            n_components = int(1)
        else:
            impossible_dimensions = True

    else:
        impossible_dimensions = True

    if impossible_dimensions:
        raise ValueError("Template (shape: %s) and data (shape: %s) dimensions are not compatible!" %
                (templates.shape, data.shape))
   
    n_samples_template = templates.shape[-1]
    if templates.shape != (n_templates, n_stations, n_components, n_samples_template):
        templates = templates.reshape(n_templates, n_stations, n_components, n_samples_template)

    n_samples_data = data.shape[-1]
    if data.shape != (n_stations, n_components, n_samples_data):
        data = data.reshape(n_stations, n_components, n_samples_data)

    assert moveouts.shape == weights.shape, "'moveouts' must have same shape as 'weights'"

    if moveouts.shape != (n_templates, n_stations, n_components):
        if (n_templates * n_stations * n_components) / moveouts.size == n_components:
            moveouts = np.repeat(moveouts, n_components).reshape(n_templates, n_stations, n_components)
        elif (n_templates * n_stations * n_components) / moveouts.size == 1.:
            moveouts = moveouts.reshape(n_templates, n_stations, n_components)

    if weights.shape != (n_templates, n_stations, n_components):
        if (n_templates * n_stations * n_components) / weights.size == n_components:
            weights = np.repeat(weights, n_components).reshape(n_templates, n_stations, n_components)
        elif (n_templates * n_stations * n_components) / weights.size == 1.:
            weights = weights.reshape(n_templates, n_stations, n_components)

    n_corr = int((n_samples_data - n_samples_template) / step + 1)

    # compute sum of squares for templates
    sum_square_templates = np.sum(templates**2, axis=-1).astype(np.float32)
    # Create output array
    cc_sums = np.zeros((n_templates, n_corr), dtype=np.float32)
    # Ensure our input arrays can be manipulated from C/C++ and has the correct in-memory layout
    templates = np.require(templates, np.float32, requirements=NP_FIELD_REQUIREMENTS)
    sum_square_templates = np.require(sum_square_templates, np.float32, requirements=NP_FIELD_REQUIREMENTS)
    moveouts = np.require(moveouts, np.intc, requirements=NP_FIELD_REQUIREMENTS)
    data = np.require(data, np.float32, requirements=NP_FIELD_REQUIREMENTS)
    weights = np.require(weights, np.float32, requirements=NP_FIELD_REQUIREMENTS)
    cc_sums = np.require(cc_sums, np.float32, requirements=NP_FIELD_REQUIREMENTS)
    # Create memory views of input data so that we can pass along to C/C++ methods imported at the top
    cdef float[:,:,:,:] temp_view = templates
    cdef float[:,:,:] sq_temp_view = sum_square_templates
    cdef int[:,:,:] moveouts_view = moveouts
    cdef float[:,:,:] data_view = data
    cdef float[:,:,:] weights_view = weights
    cdef float[:,:] cc_view = cc_sums
    # Call backend implementation depending on desired architecture
    if arch in ('cpu', 'precise'):
        ret = matched_filter_serial(&temp_view[0, 0, 0, 0], &sq_temp_view[0, 0, 0],
                &moveouts_view[0, 0, 0], &data_view[0, 0, 0], &weights_view[0, 0, 0],
                step, n_samples_template, n_samples_data, n_templates,
                n_stations, n_components, n_corr, norm, &cc_view[0, 0])
    elif arch in ('gpu', 'sycl'):
        IF FMF2_SYCL:
            ret = matched_filter_sycl(&temp_view[0, 0, 0, 0], &sq_temp_view[0, 0, 0],
                    &moveouts_view[0, 0, 0], &data_view[0, 0, 0], &weights_view[0, 0, 0],
                    step, n_samples_template, n_samples_data, n_templates,
                    n_stations, n_components, n_corr, norm, &cc_view[0, 0])
        ELSE:
            raise RuntimeError("FMF2 library not compiled with SYCL backend!")
    else:
        raise NotImplementedError("Unknown 'arch': %s, valid: %s" % (arch, AVAILABLE_BACKENDS))
    # Check to ensure that the backend did not report any issues
    if ret != 0:
        if ret == -28:
            raise RuntimeError("Allocation failed on device, reduce data amount!")
        else:
            raise RuntimeError("FMF2 backend error: %d" % ret)
    if check_zeros:
        raise NotImplementedError("'check_zero' not yet implemented")
    return cc_sums
