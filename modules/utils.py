import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from pathlib import Path

class AudioSample(np.ndarray):
    """A subclassed np.ndarray, with added metadata.
        Pretty much copy-paste from
        https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array"""

    def __new__(cls, input_array, metadata={}):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.metadata = metadata
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # See https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
        if obj is None: return
        self.metadata = getattr(obj, 'metadata', None)

    def __repr__(self):
        metadata_str = ' '.join([f"{k}={v}" for k,v in self.metadata.items()])
        info_str = f"{np.array2string(self)} dtype={self.dtype} {metadata_str}".strip()
        return f"AudioSample({info_str})"


def plot_spec(samples, sample_rate, ax=None, vmin=-120, vmax=0, fft_params=None):
    """Plot a spectrogram for an audio signal either on existing
    axes or create new ones.

    Arguments:
        samples
            - The wave to plot the spectrogram for as a numpy array

        sample_rate
            - Sampling rate of the audio clip

        ax
            - Matplotlib axes which is used for plotting (default: None)

        vmin, vmax
            - min and max values used for the scale of the colormap. Pass
            'None' to enable auto-scaling (default: -120, 0)
    """
    if fft_params is None:
        nperseg = 1024
        fft_params = {'window': signal.get_window(('tukey', 0.25), nperseg),
                      'nperseg': nperseg,
                      'noverlap': nperseg // 8,
                      'nfft': nperseg,
                      'mode': 'magnitude',
                      'detrend': 'constant'}
    freqs, times, spec = signal.spectrogram(samples, sample_rate, **fft_params)
    if ax is None:
        fig = plt.figure(figsize=(18, 5))
        ax = fig.add_subplot(111)
        show_fig = True
    else:
        show_fig = False
    # We need to scale spec appropriately for plotting
    p = ax.pcolormesh(times, freqs, 10*np.log10(spec), shading='auto',
                      cmap='magma', vmin=vmin, vmax=vmax)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    if show_fig:
        fig.colorbar(p)
        plt.show()


def load_data(folder, channel=-1, globpattern="*"):
    """Loads all WAV files from a given folder. For stereo files this function keeps
    the specified channel. Assumes that all input files have the same sampling frequency.
    Notice that channel=-1 has the potential to be destructive if the phases of the channels
    are opposite.

    Arguments:
        folder
            - A pathlib.Path containing the path to the directory with the data
                files

        channel
            - The channel to keep from stereo files with 0 and 1 corresponding
                to L and R channels, respectively. Use -1 to combine both
                channels (default: -1)

        globpattern
            - A string in unix glob format. Allows filtering the data files.
                (default: "*")

    Returns:
        rates
            - The common sampling frequency

        out
            - A list containing the converted wav files as numpy arrays

        names
            - Filenames corresponding to the arrays in `out`
    """
    files = [p for p in folder.glob(globpattern) if p.suffix.lower() == '.wav']
    # wavfile.read displays a warning if metadata hasn't been stripped from the wav files
    rates, samples = zip(*[wavfile.read(f) for f in files])

    if rates.count(rates[0]) != len(rates):
        raise ValueError(f'Error: sampling frequency of each audio file should be equal.')

    if channel not in [0, 1, -1]:
        raise ValueError(f'Invalid channel {channel}. Expected: -1, 0, 1.')
    else:
        if channel == -1:
            out = [np.mean(s, axis=1) if s.ndim == 2 else s for s in samples]
        else:
            out = [s[:, channel] if s.ndim == 2 else s for s in samples]

    print(f'Found {len(files)} files @ {rates[0]} Hz in {folder.resolve()}.')
    return rates[0], out, [f.name for f in files]

def load_data_as_objs(folder, channel=-1, globpattern="*"):
    """Loads the data with load_data, but wraps them into
    AudioSample objects."""
    rate, samples, names = load_data(folder, channel, globpattern)
    return rate, [AudioSample(samples[i], metadata={"sample_rate":rate, "filename": names[i]}) for i in range(len(samples))]

def zero_crossings(arr):
    """Compute locations of the zero-crossings in arr. Note that a crossing at
    index i in the output array corresponds to the pair of indices (i, i+1) in
    the input array.

    Arguments:
        arr
            - Numpy array for which to compute the zero-crossings

    Returns:
        cross
            - Array with the zero-crossings. 0 value corresponds to no sign
                change, -1 to pos-to-neg transition and +1 to neg-to-pos
    """
    cross = np.diff(np.where(arr > 0, 1, 0))
    return cross


def zero_cross_rate(arr, window=1):
    """Compute the zero-crossing rate (ZCR) for an audio signal with a moving
    average.

    Arguments:
        arr
            - Numpy array for which to compute the ZCR

        window
            - Length of the window to use for computing the rate (i.e. the
                moving average) (default: 1)

    Returns:
        rate
            - Numpy array of same size as `arr` containing the ZCR at each
                index
    """
    z = np.abs(zero_crossings(arr))
    return np.convolve(z, np.ones(window), 'same') / window


def times_like(arr, sample_rate=96000, start=0):
    """Returns time values corresponding to sample_rate
    for the number of frames in arr starting where the first frame of arr
    corresponds to time=start/sample_rate.
    """
    return np.linspace(start/sample_rate, (start+len(arr))/sample_rate,
                       num=len(arr))
