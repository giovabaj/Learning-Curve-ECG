from scipy.signal import butter, sosfilt


def filter_ecgs(x, low_cut=2, high_cut=50, fs=500, order=1):
    """Function to filter ECG signals with a zero phase infinite impulse response bandpass filter. The filtering is done
    in batches to avoid memory issues.
    Inputs:
    - x: npy array with ECG signals in the form (n. of ECGs x channels x time points)
    - low_cut, high_cut: low and high frequencies of the passband filter
    - fs: signal frequency in Hz
    - order: order of the filter
    Output:
    - filtered signal
    """
    # Design the filter with second-order sections format (suggested by scipy)
    sos = butter(N=order, Wn=[low_cut, high_cut], btype='bandpass', fs=fs, output='sos')
    n = x.shape[0]  # number of ECGs
    batch = min(1000, n)  # Set the batch dimension
    for i in range(int(n / batch) + 1):  # loop over batches
        # compute upper and lower indices of the batch
        i_min = i * batch
        i_max = min((i + 1) * batch, x.shape[0])
        # Filtering the signal
        x[i_min:i_max, :] = sosfilt(sos, x[i_min:i_max, :], axis=1).astype('float32')
    return x
