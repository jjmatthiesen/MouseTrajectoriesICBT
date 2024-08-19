from src.utils.imports import *


def plot_trajectory_coords(x, y, title="", screen=None):
    """

    :param x: the x coordinates
    :param y: the y coordinates.
    Note that for a representative plot, they have to be inverted, since (0,0) is on the top left of a screen
    :param title: the title of the plot
    :param screen: screen size in [width, height]
    :return:
    """
    if screen is None:
        screen = []
    plt.title(title)
    if screen:
        plt.xlim(0, screen[0])
        plt.ylim(screen[1], 0)

    plt.plot(x, y, c='#000000', zorder=10)
    plt.show()


def plot_trajectory(mov, coords="client"):
    """
    plots the trajectory (mousemove)
    :param mov: current trajectory data
    :param coords: ['client'| 'page'] which coordinates should be potted.
    """
    if coords == "page":
        x_coord_row = 3
        y_coord_row = 4
    else:
        x_coord_row = 1
        y_coord_row = 2

    idx_move = np.where(mov[:, 5] == "mousemove")[0]
    x = mov[idx_move, x_coord_row]
    y = mov[idx_move, y_coord_row]
    plt.plot(x, y, c='#000000', zorder=10)


def plot_trajectory_from_tensor(tensor, length_scale=""):
    """
    :param tensor: tensor with x and y coordinates
    :param length_scale: the length scale from the Gaussian processes (GPs)
    :return:
    """
    fig, ax = plt.subplots()
    param_text = "num_points: " + str(tensor.shape[1]) + "    length_scale: " + str(length_scale)
    ax.plot(tensor[0], tensor[1], 'k-')
    # plt.subplots_adjust(right=0.75, bottom=0.18)
    ax.set_title(param_text)
    plt.show()


def plot_clicks(mov, coords):
    """
    plots the clicks (mousedown)
    :param mov: current trajectory data
    :param coords: ['client'| 'page'] which coordinates should be potted.
    """
    idx_clicks = np.where(mov[:, 5] == "mousedown")[0]
    if coords == "page":
        x_coord_row = 3
        y_coord_row = 4
    else:
        x_coord_row = 1
        y_coord_row = 2
    x_c = mov[idx_clicks, x_coord_row]
    y_c = mov[idx_clicks, y_coord_row]
    plt.scatter(x_c, y_c, marker='X', c='blue')


def plot_mouse(mov, screen=None, coords="client", trajectory=True, clicks=True, title="", save=False, show=True,
               save_path=''):
    """
    plots trajectory image where trajectory is shown as black line and clicks in blue crosses
    :param show: show plot
    :param save_path: path to save the plot
    :param save: for saving the plots
    :param clicks: showing clicks
    :param trajectory: showing trajectory
    :param title: title of te plot
    :param coords: ['client'| 'page'] which coordinates should be potted.
    page are x and y coordinated in relation to the whole page, while client related to the screen section
    :param screen: [x,y] of the view box to see trajectory in relation to screen, window, or document
    :param mov: current trajectory data
    """
    if trajectory:
        plot_trajectory(mov, coords)
    if clicks:
        plot_clicks(mov, coords)

    plt.title(title)
    plt.xlim(0, screen[0])
    plt.ylim(screen[1], 0)

    plt.plot()
    if save:
        if save_path:
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path + title + ".png")
        else:
            pathlib.Path('../../data/investigation/').mkdir(parents=True, exist_ok=True)
            plt.savefig("../../data/investigation/" + title + ".svg")
    if show:
        plt.show()
    plt.close('all')


def plot_trajectory_and_hist(curv_list, mov, index_mov):
    """
    trajectory and the histogram of angles
    :param curv_list: the list of angles
    :param mov: the movement of x and y coordinates
    :param index_mov: the indexes where the mouse moved (no pauses or scrolls)
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('trajectory (on page) and angle overview')
    coords = "client"
    ax1.plot(np.array(mov[index_mov, 3]), np.array(mov[index_mov, 4]), c='#000000', zorder=10)
    ax2.hist(curv_list)
    plt.show()


def plot_trajectory_and_features(mov, index_mov, text):
    """
    plots the trajectory and an overview os the mouse features
    :param mov: the movement of x and y coordinates
    :param index_mov: the indexes where the mouse moved (no pauses or scrolls)
    :param text: additional text on the plot
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('trajectory (on page) and features')
    coords = "client"
    ax1.plot(np.array(mov[index_mov, 3]), np.array(mov[index_mov, 4]), c='#000000', zorder=10)
    ax2.axis([0, 10, 0, 10])
    ax2.axis('off')
    ax2.text(0, 10, text, ha="left", va='top', wrap=True)
    plt.show()


def plot_performance(train_performance, test_performance, model_name, m_save_name, labels, what, save=False, show=False):
    """
    plots the training and validation performance
    :param train_performance: the train performance over all epoch
    :param test_performance: the validation performance over all epochs
    :param model_name: the name of the model (e.g. 1DCNN)
    :param m_save_name: the name for saving including all hyperparameters
    :param labels: labels for the legend
    :param what: the title
    :param save: if the plot should be saved
    :param show: if the plot should be displayed
    :return:
    """
    fig, ax = plt.subplots()
    title = what
    ax.plot(train_performance, color='teal', label=labels[0])
    ax.plot(test_performance, color='orange', label=labels[1])
    plt.xlabel('epochs')
    plt.ylabel(title)
    ax.legend()
    ax.set_title(m_save_name, fontsize=9)
    ax.set_facecolor('#f4f4f4')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='white')
    fig.set_size_inches(7, 5, forward=True)
    if save:
        pathlib.Path('../results/performance/' + model_name).mkdir(parents=True, exist_ok=True)
        plt.savefig("../results/performance/" + model_name + "/" + title + "_" + m_save_name + ".svg")
    if show:
        plt.show()
