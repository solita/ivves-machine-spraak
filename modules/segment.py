import numpy as np
from scipy import signal


def kernel(size=16, smoothing='', strength=2**2, kernel_primitive=None):
    """Computes a kernel to be used for feature extraction from similarity
    matrices. This is done based on the given `kernel_primitive` with the
    default argument corresponding to a checkerboard kernel, as explained
    in Segmentation.ipynb.

    Arguments:
        size
            - Desired size of the kernel (as a square matrix). Note that `size`
                must be divisible by `len(kernel_primitive)` (default: 16)

        smoothing
            - The default argument corresponds to no smoothing, while the value
                'gaussian' applies a Gaussian smoothing with bandwidth given by
                `strength` (default: '')

        strength
            - Bandwidth for Gaussian smoothing (default: 4)

        kernel_primitive
            - Kernel primitive which is enlargened to K. The default argument
                corresponds to checkerboard kernel (default: none)

    Returns:
        K
            - A `size` x `size` numpy array
    """
    # strength is inversely proportional to the standard deviation of the
    # smoothing and thus strength -> 0 corresponds to reduced smoothing,
    # whereas strength -> infty completely dampens the input signal (except
    # at the mean)
    if kernel_primitive is None:
        kernel_primitive = np.array([[-1, 1],
                                    [1, -1]])
    assert size % len(kernel_primitive) == 0, f'Size {size} should be divisible by the primitive length {len(kernel_primitive)}'
    kernel_scale = size // len(kernel_primitive)
    K_raw = np.kron(kernel_primitive, np.ones((kernel_scale, kernel_scale)))
    assert K_raw.shape[0] == K_raw.shape[1]
    if not smoothing:
        K = K_raw
    elif smoothing == 'gaussian':
        # create a grid of X & Y values based on which
        # we compute the Gaussian smoothing
        X = Y = np.arange(0, len(K_raw))
        XX, YY = np.meshgrid(X, Y)
        mu = len(K_raw) // 2
        bw = size / strength
        gauss = np.exp(-.5 * ((XX-mu) ** 2 + (YY-mu) ** 2) / bw ** 2)
        smoothing = gauss * np.ones((len(K_raw), len(K_raw)))
        K = K_raw * smoothing
    else:
        raise Exception("Unknown smoothing specifier.")
    return K


def novelty_fast(A, K):
    """Same as `novelty(S, K)`, but more efficient as only the convolution
    along the diagonal is computed (which is sufficient for computing the
    novelty score).
    """
    assert K.shape[0] == K.shape[1]
    assert A.shape[0] == A.shape[1]
    sz = len(K)
    adj = len(K)  # how much to cut out from the end
    A_pd = A.copy()
    novelties = np.zeros(len(A)-adj)
    for i in range(len(novelties)):
        view = slice(i, i+sz)
        A1 = A_pd[view, view].copy()
        novelties[i] = np.multiply(A1, K).sum() / (sz ** 2)

    return novelties
