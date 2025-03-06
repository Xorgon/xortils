import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def initialize_plt(font_size=10, line_scale=1, capsize=3, latex=True, dpi=300, colourblind=True, font='cmr10'):
    """
    Initialise matplotlib with some nice settings. TODO: documentation and font options

    :param font_size: Font size (points).
    :param line_scale: Scale of all lines.
    :param capsize:
    :param latex:
    :param dpi:
    :param colourblind:
    :param font:
    """
    warnings.filterwarnings("ignore", "cmr10")  # STOP TELLING ME THIS PLEASE I KNOW WHAT FONT I WANT
    if colourblind:
        sns.set_palette("colorblind")
        # plt.style.use('seaborn-colorblind')
    if (latex):
        plt.rc('text', usetex=True)
        plt.rc('text.latex',
               preamble=r'\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}\DeclareMathAlphabet\mathsfbi{OT1}{cmss}{m}{sl}')
        font = {'family': 'serif', 'size': font_size, 'serif': ['cmr10']}
        plt.rc('font', **font)
    plt.rc('lines', linewidth=line_scale, markersize=3 * line_scale)
    plt.rc('xtick.major', width=line_scale / 2)
    plt.rc('xtick.minor', width=line_scale / 2)
    plt.rc('ytick.major', width=line_scale / 2)
    plt.rc('ytick.minor', width=line_scale / 2)
    plt.rc('axes', linewidth=0.5 * line_scale)
    plt.rc('patch', linewidth=0.5 * line_scale)
    plt.rc('figure', dpi=dpi, autolayout=True)
    plt.rc('errorbar', capsize=capsize)
