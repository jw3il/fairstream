import shutil
import matplotlib
import matplotlib.colors as mcolors

# use tex
if shutil.which("latex"):
    matplotlib.rcParams["text.usetex"] = True
else:
    print("WARNING: LaTeX installation not found!")

COLORS = mcolors.TABLEAU_COLORS
COLORS_LIST = [col for (key, col) in COLORS.items()]

# export with higher dpi
matplotlib.rcParams["savefig.dpi"] = 300

# switch to True for high res 3d surface plots (will take longer)
SURFACE_PLOT_HIGH_TRI_COUNT = True


def set_matplotlib_font_size(small_size=8, medium_size=10, bigger_size=12):
    """
    Sets matplotlib font sizes, see
    https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot.

    :param small_size: size for font, axes title, xtick, ytick, legend
    :param medium_size: size for axes label
    :param bigger_size: size for figure title
    """
    import matplotlib.pyplot as plt

    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small_size)  # legend fontsize
    plt.rc("figure", titlesize=bigger_size)  # fontsize of the figure title


def init_plt(stye=True):
    import matplotlib.pyplot as plt

    #if "ggplot" in plt.style.available:
    #    plt.style.use("ggplot")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.grid.axis"] = "y"
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["legend.frameon"] = "True"
    plt.rcParams["legend.framealpha"] = "1.0"
    plt.rc("font", **{"family": "serif"})
    # http://phyletica.org/matplotlib-fonts/
    # not necessary as we use tex
    # plt.rcParams['pdf.fonttype'] = 42
    # plt.rcParams['ps.fonttype'] = 42


def adjust_lightness(color, amount=0.5):
    """
    Color lightness adjustment by Ian Hincks
    See https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def adjust_lightness_relative(color, alpha):
    """
    Color lightness adjustment by Ian Hincks
    See https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 * (1 - alpha) + alpha * c[1], c[2])
