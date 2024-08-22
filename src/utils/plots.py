from src.utils.imports import *
from sklearn.metrics import RocCurveDisplay, auc
import json
import statsmodels.stats.api as sms

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


def substract_small(dropouts: pd.Series, i: int) -> float:
    """
    Calculate the cumulative percentage of participants remaining after a certain number of modules.

    Args:
        dropouts (pd.Series): Series representing the percentage of dropouts at each module.
        i (int): Number of modules to consider for the calculation.

    Returns:
        float: The cumulative percentage of participants remaining after i modules.
    """
    # Sort the dropout series by its index to ensure correct order
    dropouts = dropouts.sort_index()
    value = 100  # Start with 100% of participants

    # Subtract dropout percentages for each module up to module i
    for j in range(0, i):
        value -= dropouts[j]

    return value


def cum_dropout(df: pd.DataFrame, int_arm: str, color: str) -> None:
    """
    Plot the cumulative dropout curve for a given treatment arm.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        int_arm (str): The treatment arm for which to calculate the dropout curve.
        color (str): Color to use for the plot line.

    Returns:
        None
    """
    # Calculate the percentage of dropouts for each module in the specified treatment arm
    dropouts = df[df['Treatment_norm'] == int_arm]['Antal moduler'].value_counts(normalize=True) * 100

    # Ensure all module indices from 0 to 9 are present in the dropout series, filling missing indices with 0%
    for mis in [inds for inds in range(0, 10) if inds not in dropouts.index]:
        dropouts.loc[mis] = 0

    # Plot the cumulative dropout curve
    plt.plot([i for i in range(0, len(dropouts))],
             [substract_small(dropouts, i) for i in range(0, len(dropouts))],
             label=int_arm, c=color)


def plot_all_dropout(df1: pd.DataFrame, cut_off_dropout_sophia: int, cut_off_dropout_dana: int) -> None:
    """
    Plot the dropout curves for all treatment arms and highlight cutoff points.

    Args:
        df1 (pd.DataFrame): DataFrame containing the dataset.
        cut_off_dropout_sophia (int): Module number indicating the cutoff for the Sophia treatment.
        cut_off_dropout_dana (int): Module number indicating the cutoff for the Dana treatment.

    Returns:
        None
    """
    # Define colors for each treatment arm
    colors = {
        'Dana': 'darkred',
        'Panic': 'gold',
        'Social_Anxiety': 'darkgreen',
        'Depression': 'darkblue'
    }

    plt.figure(figsize=(7, 5))

    # Plot the dropout curve for each treatment arm
    for arm in colors.keys():
        cum_dropout(df1, arm, colors[arm])

    # Add labels, lines, and annotations to the plot
    plt.xlabel('Number of sessions completed')
    plt.ylabel('% of users in the intervention')
    plt.vlines(cut_off_dropout_sophia, ymin=0, ymax=100, colors='grey', linestyles='dashed')
    plt.vlines(cut_off_dropout_dana, ymin=0, ymax=100, colors='darkgrey', linestyles='dashed')
    plt.hlines(df1.dropout_mod.mean() * 100, xmin=0, xmax=10, colors='black', linestyles='dashed')
    plt.text(10.3, df1.dropout_mod.mean() * 100, 'Avg.')
    plt.text(10.1, df1.dropout_mod.mean() * 91, 'Dropout')
    plt.legend(['Dana Depression', 'Sophia Panic', 'Sophia Social Anxiety', 'Sophia Depression', 'Cutoff Sophia',
                'Cutoff Dana'])
    plt.grid(False)
    plt.show()


def nested_auccurve_nonseq(results: dict, desc: str, name: str, k: int, reps: int) -> list[float]:
    """
    Plot ROC curves for non-sequential cross-validation results and calculate the AUCs.

    Args:
        results (dict): Dictionary containing the results of cross-validation.
        desc (str): Description for the plot title.
        name (str): Name for the plot title and file output.
        k (int): Number of folds in the cross-validation.
        reps (int): Number of repetitions in the cross-validation.

    Returns:
        list[float]: A list of AUC scores from the cross-validation.
    """
    fig, ax = plt.subplots()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)  # Interpolated mean false positive rate

    # Iterate over repetitions to generate ROC curves
    for j in range(0, reps):
        y_pred = [y_preds for i in range(0, k) for y_preds in results[j][i]['y_proba']]
        y_true = [y_true for i in range(0, k) for y_true in results[j][i]['y_true']]

        # Plot ROC curve for each repetition
        viz = RocCurveDisplay.from_predictions(y_true, y_pred,
                                               name=f'{j}',
                                               alpha=0.3, lw=1, ax=ax)
        # Interpolate true positive rate and collect AUC
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    # Calculate mean and standard deviation for TPR and AUC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plot the mean ROC curve
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean (AUC: %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # Plot the standard deviation range
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std.')

    # Finalize and save the plot
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=f"{desc} ROC Curves Cross-Validation {name}")
    ax.legend(loc="lower right")
    fig.savefig(f'../results/graphs/{desc}{name}_roc_nested_.png')
    plt.show()

    # Calculate and log 95% confidence interval for AUC
    ci = sms.DescrStatsW(aucs).tconfint_mean()
    logging.info(f'Average outer score {mean_auc} with SD {std_auc} and 95%-CIs {ci[0]:.2f},{ci[1]:.2f}')

    return aucs


def nested_auccurve_nn(results: dict, desc: str, name: str, k: int) -> list[float]:
    """
    Plot ROC curves for neural network cross-validation results and calculate the AUCs.

    Args:
        results (dict): Dictionary containing the results of cross-validation.
        desc (str): Description for the plot title.
        name (str): Name for the plot title and file output.
        k (int): Number of folds in the cross-validation.

    Returns:
        list[float]: A list of AUC scores from the cross-validation.
    """
    fig, ax = plt.subplots()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)  # Interpolated mean false positive rate

    # Iterate over folds to generate ROC curves
    for i in range(0, k):
        viz = RocCurveDisplay.from_predictions(results[i]['y_true'], results[i]['y_proba'],
                                               name=f'{i}',
                                               alpha=0.3, lw=1, ax=ax)

        # Interpolate true positive rate and collect AUC
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    # Calculate mean and standard deviation for TPR and AUC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plot the mean ROC curve
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean (AUC: %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # Plot the standard deviation range
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std.')

    # Finalize and save the plot
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=f"{desc} ROC Curves Cross-Validation {name}")
    ax.legend(loc="lower right")
    fig.savefig(f'../results/graphs/{desc}{name}_roc_nested_.png')
    plt.show()

    # Calculate and log 95% confidence interval for AUC
    ci = sms.DescrStatsW(aucs).tconfint_mean()
    logging.info(f'Average outer score {mean_auc} with SD {std_auc} and 95%-CIs {ci[0]:.2f},{ci[1]:.2f}')

    return aucs


def graph_nns(path_to_jsn: str, k: int, desc: str, name: str) -> None:
    """
    Load results from a JSON file, process them, and plot the ROC curves for neural networks.

    Args:
        path_to_jsn (str): Path to the JSON file containing the results.
        k (int): Number of folds in the cross-validation.
        desc (str): Description for the plot title.
        name (str): Name for the plot title and file output.

    Returns:
        None
    """
    # Load data from JSON file
    with open(path_to_jsn) as json_data:
        data = json.load(json_data)

    results = {}

    # Process data for each fold
    for fold in range(0, k):
        df2 = pd.DataFrame.from_dict(data[str(fold)], orient="index").reset_index()
        results[fold] = {
            'y_proba': [y_preds for i in range(0, 5) for y_preds in df2.loc[i, 'predict_probas']],
            'y_true': [y_trues for i in range(0, 5) for y_trues in df2.loc[i, 'y_true']]
        }

    # Generate and plot the nested AUC curve
    nested_auccurve_nn(results, desc, name, k)