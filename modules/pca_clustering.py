from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import CenteredNorm, Normalize
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import librosa


def split_spec(spec, window, stride=1, verbose=False):
    """Construct segments from the frames of a spectrogram according to weights
    given in `window`. The windows can be overlapping or disjoint, which is
    controlled by `stride`. Using `stride=len(window)` recovers the old
    behaviour of disjoint adjacent windows. Notice no aggregation is done
    within the segments (unlike with typical filtering in signal processing).
    The output is a matrix with each row corresponding to one of the flattened
    segments. The input data is padded (minimally) so that it can be covered
    evenly by windows with the given stride.

    Arguments:
        spec
            - Spectrogram from scipy.signal.spectrogram that we want to
                transform

        window
            - Numpy array of weights (from e.g. scipy.signal.get_window()) to
                use for segmenting

        stride
            - Number of frames by which the window is moved forward after each
                step (default: 1)

        verbose
            - Whether to print some debug info (default: False)

    Returns:
        X
            - Matrix of the resulting segments
    """
    assert isinstance(stride, int) and stride > 0, "stride needs to be a positive integer"
    frames = spec.shape[1]
    freqs = spec.shape[0]
    wdw_len = len(window)
    padding = -(frames - wdw_len) % stride
    s = np.zeros((frames + padding, freqs))
    s[:frames, :] = spec.T.copy()
    steps = (frames + padding - wdw_len) / stride
    assert steps.is_integer()
    steps = int(steps) + 1  # +1 for the initial step
    X = np.zeros((steps, wdw_len * freqs))
    if verbose:
        print(f"padding={padding}")
        print(f"steps={steps}")
    for i in range(steps):
        # flatten so we make a copy
        X[i, :] = (s[i * stride:i * stride + wdw_len, :]
                   * window[:, np.newaxis]).flatten()
    return X


def plot_pca(spec_pca, times, stride=1, ax=None, halfrange=0):
    """Plots the size of the components of a (PCA) projected spectrogram as
    a heatmap over time. Shows at most the first 10 components.

    Arguments:
        spec_pca
            - Projected data to plot (from spec2pca)

        times
            - Values for the x-axis (time)

        stride
            - The stride parameter used to subdivide the original spectrogram
                (default: 1)

        ax
            - If not None, then draw the graph on this axis (default: None)

        halfrange
            - If greater than 0, then use this to specify the max absolute
                distance from 0 for the purpose of scaling the colormap.
                Otherwise determined from input data (default: 0)

    Returns:
        fig
            - Only returned if ax was not provided. The figure used for the
                plot

        ax
            - Only returned if ax was not provided. The ax corresponding to fig
    """
    # To roughly match the original audio length we recover the "unstrided"
    # resolution first
    s_strided = np.repeat(spec_pca, repeats=stride, axis=1)
    if ax is None:
        fig = plt.figure(figsize=(18, 5))
        ax = fig.add_subplot(111)
        return_fig = True
    else:
        return_fig = False
    # Show at most the first 10 principal components
    n_components_to_show = min(s_strided.shape[0], 10)
    max_len = min(len(times), s_strided.shape[1])
    Z = s_strided[:n_components_to_show, :max_len]
    if halfrange > 0:
        cnorm = CenteredNorm(halfrange=halfrange)
    else:
        cnorm = CenteredNorm()
    p = ax.pcolormesh(times[:max_len], np.arange(1, n_components_to_show+1), Z,
                      norm=cnorm, shading='auto', cmap='PRGn')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))
    ax.set_ylabel('PC')
    ax.set_xlabel('Time (s)')
    if return_fig:
        if halfrange > 0:
            norm = Normalize(vmin=-halfrange, vmax=halfrange)
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap='PRGn'))
        else:
            fig.colorbar(p)
        return fig, ax


def plot_pca_grid(spec_pcas, ts, stride=1, names=[], title='', hl_idx=[],
                  hl_label='', equal_scale=True):
    """Plots all the projected spectrograms as a grid (assumes 8 samples at
    the moment) by using plot_pca. Optionally highlight certain indices.

    Arguments:
        spec_pcas
            - List of the projected spectrograms to plot

        ts
            - List of the corresponding x-axis values (time)

        stride
            - Used to match the time stamps of the projected data to
                the original. See `plot_pca` (default: 1)

        names
            - List of titles for each individual plot

        title
            - A caption for the whole figure

        hl_idx
            - List of indices to highlight

        hl_text
            - If non-empty, then used as a label for the legend for
                the highlighted plots

        equal_scale
            - Whether to normalise the colormaps uniformly across each plot
                (default: True)

    Returns:
        fig
            - The figure for the full picture
    """
    hl_color = 'coral'
    hr = 0
    if equal_scale:
        for s in spec_pcas:
            hr = max(hr, abs(s.min()), abs(s.max()))

    nclips = len(spec_pcas)
    ncols = 2
    nrows = ceil(nclips/ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(14, 8))
    fig.tight_layout(pad=1.5, rect=[0, 0.03, 1, 0.95])

    p = None
    for n, ax in enumerate(axs.flat):
        if n >= nclips:
            fig.delaxes(ax)
        else:
            p = plot_pca(spec_pcas[n], ts[n], stride=stride, ax=ax,
                         halfrange=hr)
            if len(names) > 0:
                ax.set_title(names[n])
            if n < len(axs.flat) - 2:
                ax.set_xlabel('')
            if n % 2 == 1:
                ax.set_ylabel('')
            if n in hl_idx:
                ax.tick_params(color=hl_color, labelcolor=hl_color)
                for spine in ax.spines.values():
                    spine.set_edgecolor(hl_color)

    if hl_idx and hl_label:
        fig.legend(handles=[Patch(facecolor=hl_color, label=hl_label)],
                   loc='center right')

    if equal_scale:
        fig.subplots_adjust(right=0.93)
        cbar_ax = fig.add_axes([0.95, 0.4+(0.2/3)*2, 0.01, 0.4])
        norm = Normalize(vmin=-hr, vmax=hr)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='PRGn'),
                     cax=cbar_ax)

    if title:
        fig.suptitle(title)
    return fig


def sample2spec(samples, sample_rate, tfm='', fft_params=None):
    """Convert a list of audio samples into Scipy spectrograms. Notice that
    scipy spectrogram matrices are of the form (freqs x times). Since the
    frequencies are our features, we have to take the transpose of the
    spectrogram if we want to have the data in standard form to be used with
    e.g. sklearn.

    Arguments:
        samples
            - List of samples (as numpy arrays) to convert

        sample_rate
            - Sample rate of the samples (same for each sample)

        tfm
            - Optionally apply `tfm` to resulting Fourier decompositions

    Returns:
        freqs
            - List of frequencies (i.e. the y-axis for the spectrogram) for
                each sample

        times
            - List of time steps (i.e. the x-axis for the spectrogram) for
                each sample

        specs
            - List of spectrograms from signal.spectrogram for each sample
    """
    freqs, times, specs = [], [], []
    if not tfm:
        tfm = lambda x: x
    elif tfm == 'log':
        tfm = np.log10

    if fft_params is None:
        nperseg = 1024
        fft_params = {'window': signal.get_window(('tukey', 0.25), nperseg),
                      'nperseg': nperseg,
                      'noverlap': nperseg // 8,
                      'nfft': nperseg,
                      'mode': 'magnitude',
                      'detrend': 'constant'}
    for s in samples:
        f, t, Sxx = signal.spectrogram(s, sample_rate, **fft_params)
        freqs.append(f)
        times.append(t)
        specs.append(tfm(Sxx))
    print(f'STFT window is {fft_params["nperseg"] / sample_rate:.3f} s.')
    return freqs, times, specs


def spec2pca(Sxx, window, stride=1, n_components=10, normalise=True, sclr=None,
             pca=None):
    """Reduce the dimension of spectrograms by projecting them to axes of
    principal components. The principal axes can either be provided or computed
    from scratch. The input features to PCA are computed with `split_spec`
    according to the given window and stride.

    Arguments:
        Sxx
            - The raw spectrograms which will be projected

        window
            - The weights used for windowing the spectrogram. See split_spec.
                As a byproduct this results in downsampled end result while
                still maintaining maximum input granularity for PCA

        stride
            - The gap between the initial frames of each windowed segment.
                See split_spec (default: 1)

        n_components
            - Number of principal components to keep (default: 10)

        normalise
            - Whether to standardise the spectrograms individually
              (frequency-wise) prior to fitting PCA (default: True)

        sclr
            - An instance of one of the sklearn scalers e.g. (`StandardScaler`)
                which is used to normalise the data prior to fitting PCA. If
                one is not provided, `sklearn.preprocessing.StandardScaler`
                is fit. To bypass you can pass an identity `TransformerMixin`
                subclass

        pca
            - If not provided, fit a new PCA model to the input data. Otherwise
                use this (default: None)

    Returns:
        spec_pcas
            - List of the projected spectrograms

        pca
            - Either the original pca instance or the one created in this
                function

        sclr
            - Either the original scaler instance or the `StandardScaler`
                created in this function

        clip_scalers
            - A list of `StandardScaler` instances used to normalise each
              individual spectrogram prior to windowing. This mainly needed
              for debugging and analysing the results, not for any algorithm.
    """
    specs = [s.copy() for s in Sxx]
    # make sure each clip is standardised individually
    clip_scalers = []
    if normalise:
        for i in range(len(specs)):
            s = specs[i]
            scl = StandardScaler().fit(s.T)
            clip_scalers.append(scl)
            specs[i] = scl.transform(s.T).T

    # Xs is a list of windowed segments (as a matrix) for each input clip
    Xs = [split_spec(s, window, stride=stride) for s in specs]
    # X is what we feed into PCA, need to standardise it too
    X = np.vstack(Xs)
    if sclr is None:
        print("Fitting new scaler")
        sclr = StandardScaler().fit(X)
    X = sclr.transform(X)
    # We need the scaled Xs for individual clips
    # This is lazy and could just be recovered from X without extra computation
    Xs = [sclr.transform(x) for x in Xs]

    if not isinstance(pca, PCA):
        pca = PCA(n_components=n_components, random_state=42).fit(X)

    # would need to repeat with repeats=stride to match original audio length,
    # but this is only done when visualising the data
    spec_pca = [pca.transform(x).T for x in Xs]

    return spec_pca, pca, sclr, clip_scalers


def cluster(samples, sample_rate, window, stride=1, n_clusters=3,
            n_components=10, **kwargs):
    """Perform k-means clustering on PCA-projected spectrograms of the input audio
    samples (as a list of numpy arrays corresponding to waveforms).

    Arguments:
        samples
            - List of audio samples to cluster

        sample_rate
            - The common sampling rate of the audio clips

        window
            - List of weights to use for windowing the spectrograms
                (see spec2pca)

        stride
            - How much the window is moved for each segment (see spec2pca).
                Essentially a downsampling factor for time dimension
                (default: 1)

        n_clusters
            - The number of clusters to fit

        n_components
            - The number of principal components to use

        **kwargs
            - Passed to the KMeans object

    Returns:
        spec_pcas
            - List of the projected spectrograms

        ts
            - List of times for the x-axis of each projection

        X_pca
            - The final data matrix used for k-means (this is mainly for
                debugging)

        kmeans
            - The trained KMeans object

        pca
            - The fitted PCA object

        kmeans_sclr
            - The scaling sklearn transformer used prior to fitting KMeans

        pca_sclr
            - The scaling sklearn transformer used prior to fitting PCA
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)

    fs, ts, Sxx = sample2spec(samples, sample_rate)

    spec_pcas, pca, pca_sclr, _ = spec2pca(Sxx, window, stride=stride,
                                           n_components=n_components,
                                           normalise=True)

    X_pca = np.hstack(spec_pcas).T
    kmeans_sclr = StandardScaler().fit(X_pca)
    X_pca = kmeans_sclr.transform(X_pca)

    kmeans.fit(X_pca)

    return spec_pcas, ts, X_pca, kmeans, pca, kmeans_sclr, pca_sclr


def spec2mel(Sxx, sample_rate, n_mels=128, **kwargs):
    """Convert an existing spectrogram to Mel scale.
    """
    D = Sxx ** 2
    xx = librosa.feature.melspectrogram(S=D, sr=sample_rate, n_mels=n_mels,
                                        **kwargs)
    return xx


def spec2mfcc(Sxx, sample_rate, n_mels=128, n_mfcc=20, **kwargs):
    """Extract the MFCCs of an existing spectrogram.
    """
    D = Sxx ** 2
    xx = librosa.feature.melspectrogram(S=D, sr=sample_rate, n_mels=n_mels,
                                        **kwargs)
    xx = librosa.feature.mfcc(S=xx, sr=sample_rate, n_mfcc=n_mfcc)
    return xx


def spec2ceps(specs, LB=0, UB=None):
    """Compute cepstrums for a given list of spectrograms. Retain the
    coefficients from LB to UB. UB=None corresponds to no upper bound.

    Returns:
        qfs
            - A list of the corresponding quefrencies (just for compatibility,
                                                       not accurate values atm)
        ceps
            - The cepstral coefficients of the absolute cepstrum

        cceps
            - The cepstral coefficients of the complex cepstrum (main useful
            for if we want to invert the process).
    """
    assert LB >= 0
    ceps = []
    cceps = []
    qfs = []
    for spec in specs:
        lp = 2*np.log10(np.abs(spec))
        cp = np.fft.rfft(lp, axis=0)
        # cp = fft.dct(lp, axis=0)
        if UB is None:
            cp = cp[LB:]
        else:
            cp = cp[LB:UB]
        cceps.append(cp)
        ceps.append(np.abs(cp))
# fix to correct quefs based on sample_rate & fft_params
        qfs.append(np.arange(len(cp)))
    return qfs, ceps, cceps


def spec2erg(specs, sample_rate, tfm='', fft_params=None):
    """Compute the Fourier transform of the log-spectral energy for the
    spectrograms in specs.
    """
    lp = [2*np.log(np.linalg.norm(sx, ord=2, axis=0)) for sx in specs]
    fs, ts, erg = sample2spec(lp, sample_rate, tfm, fft_params)
    return fs, ts, erg
