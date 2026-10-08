import matplotlib.pyplot as plt

def set_plot_style():
    # Set figure size
    plt.rcParams['figure.figsize'] = (6, 6)

    # Set font family and size
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['mathtext.it'] = 'Arial:italic'
    plt.rcParams['mathtext.rm'] = 'Arial'
    plt.rcParams['mathtext.tt'] = 'Arial'
    plt.rcParams['mathtext.bf'] = 'Arial:bold'
    plt.rcParams['mathtext.cal'] = 'Arial'
    plt.rcParams['mathtext.sf'] = 'Arial'
    plt.rcParams['mathtext.fontset'] = 'custom'

    # Set axis line width
    plt.rcParams['axes.linewidth'] = 1

    # Set tick line width
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1

    # Other settings
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.color'] = '#000000'