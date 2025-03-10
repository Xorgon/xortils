import numpy as np
import scipy.fft as fft
from scipy.ndimage import uniform_filter1d
import scipy.signal as sig


def do_fft(signal, sample_period, detrend=None,
           filter_type=None, window_size=1, window_width=0.05, samples_per_window=20):
    """
    Perform an FFT analysis on a signal. This returns only the positive frequencies and will optionally filter the data.

    :param signal: Array of data.
    :param sample_period: Sample period of data (in seconds).
    :param detrend: Type of detrending applied. Options: None, 'linear', 'constant'
        (default: None - see scipy.signal.detrend).
    :param filter_type: Optional filter type. Options: None, 'uniform', 'decade'
    :param window_size: Window size for uniform filter when ``filter_type`` is 'uniform' (default: 1 - no filtering).
    :param window_width: Window width for decade filter when ``filter_type`` is 'decade'. See also ``decade_filter``.
    :param samples_per_window: Samples per window for decade filter when ``filter_type`` is 'decade'.

    :return: frequencies, fft_amplitudes
    """
    if detrend is not None:
        sig.detrend(signal, type=detrend, overwrite_data=True)

    fft_out = fft.fft(signal)
    fft_freqs = fft.fftfreq(len(signal), d=sample_period)
    fft_pro = np.abs(fft_out[:round(len(fft_freqs) / 2)])
    pos_freqs = fft_freqs[:round(len(fft_freqs) / 2)]

    match filter_type:
        case 'uniform':
            filt_freqs = pos_freqs
            filt_fft = uniform_filter1d(fft_pro, window_size)
        case 'decade':
            filt_freqs, filt_fft = decade_filter(pos_freqs, fft_pro, window_width, samples_per_window)
        case _:
            filt_freqs, filt_fft = pos_freqs, fft_pro

    return filt_freqs, filt_fft


def decade_filter(frequencies, fft_amplitudes, window_width=0.05, samples_per_window=20, interp_to_freqs=False):
    """
    Filter an FFT using a log-space moving average ("decade filter"?). The window size of the moving average is defined
    in logarithmic frequency as ``window_width``.

    For efficiency, this function also down-samples the resulting FFT to avoid excessive sampling of the smooth FFT.
    The smoothed FFT can optionally be up-sampled by interpolation back to the original frequencies. This is controlled
    with the ``interp_to_freqs`` parameter.

    :param frequencies: Frequencies associated with each FFT amplitude.
    :param fft_amplitudes: FFT amplitudes to filter.
    :param window_width: Logarithmic window width (default: 0.1 - "one-tenth decade filter").
    :param samples_per_window: The number of output samples per window width.
    :param interp_to_freqs: Boolean, whether to interpolate the output to match with the original ``frequencies``.
    :return: filtered_freqs, filtered_fft

    **Notes**:

    The moving average is clipped to the boundaries of the FFT.

    When the window edges do not fall at an exact integer value, the window edge values are linearly interpolated from
    the two nearest values.
    """
    filtered_freqs, filtered_fft, sampled = [], [], []  # Prepare sampling arrays.

    # Determine indices at which to sample the filtered FFT.
    sample_idxs = np.round(10 ** np.arange(-1, np.log10(len(fft_amplitudes) - 1),
                                           window_width / samples_per_window)).astype(int)

    for n in sample_idxs:  # Loop through the sample indices
        if n in sampled:  # Skip any indices that we already sampled.
            continue

        # Find the limits of the window.
        float_min_idx = n * 10 ** -(window_width / 2)
        float_max_idx = n * 10 ** (window_width / 2)

        ################################
        # Compute sum of central chunk #
        ################################
        cc_min_idx = int(np.max([np.ceil(float_min_idx), 0]))
        cc_max_idx = int(np.min([np.floor(float_max_idx), len(fft_amplitudes) - 1]))
        cc_sum = np.sum(fft_amplitudes[cc_min_idx:cc_max_idx + 1])

        #################
        # Compute edges #
        #################
        if cc_min_idx == 0:  # There isn't a minimum edge because we're at the lower limit already
            min_edge = None
        else:
            min_edge = (float_min_idx % 1) * fft_amplitudes[cc_min_idx - 1] + \
                       (1 - float_min_idx % 1) * fft_amplitudes[cc_min_idx]

        if cc_max_idx == len(fft_amplitudes) - 1:  # There isn't a maximum edge because we're at the upper limit already
            max_edge = None
        else:
            max_edge = (float_max_idx % 1) * fft_amplitudes[cc_max_idx] + \
                       (1 - float_max_idx % 1) * fft_amplitudes[cc_max_idx + 1]

        ##################
        # Sum everything #
        ##################

        # Central chunk first
        total_window = cc_max_idx + 1 - cc_min_idx
        total_sum = cc_sum

        # Add minimum edge
        if min_edge is not None:
            edge_width = 1 - (float_min_idx % 1)
            total_window += edge_width
            total_sum += min_edge * edge_width  # Weight by the width of the edge

        # Add maximum edge
        if max_edge is not None:
            edge_width = float_max_idx % 1
            total_window += edge_width
            total_sum += max_edge * edge_width  # Weight by the width of the edge

        filtered_freqs.append(frequencies[n])  # Store down-sampled frequencies.
        filtered_fft.append(total_sum / total_window)  # Take average over the window and store it.
        sampled.append(n)  # Store samples so we don't sample here again.

    if interp_to_freqs:
        filtered_fft = np.interp(frequencies, filtered_freqs, filtered_fft)
        return frequencies, filtered_fft
    else:
        return np.array(filtered_freqs), np.array(filtered_fft)
