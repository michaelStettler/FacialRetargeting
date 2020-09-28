import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from plotting import plot_cumulative_correlations


def low_pass_filter(signal, ksize=3, theta=1):
    """
    apply low pass filter with a gaussian kernel of size ksize and theta

    t:= length of signal

    :param signal: input 1D signal (t,)
    :param ksize: size of Gaussian kernel
    :param theta: variance of Gaussian
    :return: filtered signal
    """
    # built 1D gaussian filter
    x = np.arange(ksize) - int(ksize/2)
    filt = np.exp(-x**2/(2*theta**2))/np.sqrt(2*np.pi*theta**2)

    # filter signal
    return np.convolve(signal, filt, mode='same')


def get_filt_signals(signals, ksize=3, theta=1):
    """
    Apply a 1D low pass filter over each row of the input: signals

    k:= number of different signals
    t:= length of signals

    :param signals: input 2D signals (k, t)
    :param ksize: size of 1D Gaussian kernel
    :param theta: variance of Gaussian
    :return:
    """
    filt_signals = np.zeros(np.shape(signals))
    for i in range(np.shape(signals)[0]):
        signal = signals[i]
        filt_signal = low_pass_filter(signal, ksize, theta)
        filt_signals[i] = filt_signal

    return filt_signals


def get_key_expressions(sequence, ksize=3, theta=1, do_plot=False):
    """
    Extract key expressions as in 4.2 Key Expression Extraction

    k:= number of blendshapes
    f:= number of frames

    :param sequence: input data (k, f)
    :param ksize: int parameter to define the size of the 1D Gaussian kernel size
    :param theta: float parameter to define the Gaussian filter
    :param do_plot: option to plot the cumulative correlation as in Fig. 8
    :return: key expressions within sequence
    """
    # apply low band filtering over each row
    filtered_seq = get_filt_signals(sequence, ksize, theta)

    # sum filtered correlations coefficients over column
    cumul_seq = np.sum(filtered_seq, axis=0)

    # extract local peaks
    key_expressions, _ = find_peaks(cumul_seq)

    if do_plot:
        plot_cumulative_correlations(cumul_seq, key_expressions)

    return sequence[:, key_expressions]


if __name__ == '__main__':
    """
    Built several test cases to try and plot the effect of the three functions:
        - low_pass_filter
        - get_filt_signals
        - get_key_expressions
        
    run: python -m utils.get_key_expressions
        
    """
    np.random.seed(0)
    print("--------- test 1D filtering ----------")
    # test low_pass_filter
    # create random noise signals
    f = 5
    sample = 800
    n_features = 2
    x = np.linspace(0, 1, sample)
    noise = 0.08 * np.random.normal(0, 1, size=sample)
    signal = np.sin(2 * np.pi * f * x) + noise

    # apply low pass
    filt_signal = low_pass_filter(signal, ksize=3, theta=3)

    # plot signals
    plt.figure()
    plt.title("1D signals")
    plt.plot(x, signal, '-b', label='sig 0')
    plt.plot(x, filt_signal, '-r', label='sig 0 filt')
    plt.legend()

    print()
    print("--------- test k-size 1D filtering ----------")
    # test get_filt_signals function
    k = 5
    assert k >= 3  # make sure k is bigger than 3 for plotting
    # build k phase-shifted signals
    signals = np.zeros((5, sample))
    for i in range(k):
        signals[i] = np.sin(2 * np.pi * f * x + (i*2*np.pi)/k) + noise

    # apply 1D low pass filter over each k signals
    filt_signals = get_filt_signals(signals, ksize=3, theta=3)

    # plot signals
    x = np.repeat(np.expand_dims(x, axis=1), k, axis=1).T
    plt.figure()
    plt.title("k={0} phase-shifted signals".format(str(k)))
    plt.plot(x[0], signals[0], '-b', label='signals 0')
    plt.plot(x[0], filt_signals[0], '--r', label='signals 0')
    plt.plot(x[1], signals[1], '-y', label='signals 1')
    plt.plot(x[1], filt_signals[1], '--r', label='signals 1')
    plt.plot(x[2], signals[2], '-g', label='signals 2')
    plt.plot(x[2], filt_signals[2], '--r', label='signals 2')
    plt.legend()

    print()
    print("--------- test cumuluative extraction ----------")
    # test get_key_expressions function
    sequence = signals
    get_key_expressions(sequence, ksize=3, theta=2, do_plot=1)

    plt.show()