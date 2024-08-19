from src.utils.imports import *
from src.utils.imports_nn import *


def get_euc_dist(coord_x1, coord_y1, coord_x2, coord_y2):
    """
    calculates the Euclidean distance between two points
    :param coord_x1: x coordinate of point one
    :param coord_y1: y coordinate of point one
    :param coord_x2: x coordinate of point two
    :param coord_y2: y coordinate of point two
    :return: the Euclidean distance
    """
    a = coord_x1 - coord_x2
    b = coord_y1 - coord_y2
    euc_dist = math.sqrt(a * a + b * b)
    return euc_dist


def get_time_diff_list(mov, x_row_idx, y_row_idx):
    """
    :param mov: all coordinates, where the mouse moved
    :param x_row_idx: the index (number) of the column with the x values
    :param y_row_idx: the index (number) of the column with the y values
    :return:
    """
    time_diff_list = []
    for i in range(1, len(mov)):
        x_coord = mov[i][x_row_idx]
        x_coord_prev = mov[i - 1][x_row_idx]
        y_coord = mov[i][y_row_idx]
        y_coord_prev = mov[i - 1][y_row_idx]
        dist = get_euc_dist(x_coord_prev, y_coord_prev, x_coord, y_coord)
        time_diff = 0.00001 if mov[i][0] - mov[i - 1][0] == 0 else mov[i][0] - mov[i - 1][0]

        time_diff_list.append(time_diff)
    return time_diff_list


def transform_to_ints(string_array):
    """
    :param string_array: array of strings of numbers
    :return: array of integers
    """
    return [int(i) for i in string_array]


def transform_to_torch_list(li):
    """
    :param li: a list of lists
    :return: a list of torch tensor
    """
    torch_list = []
    for el in li:
        torch_list.append(torch.tensor(el.values, dtype=torch.float32))
    return torch_list


def standard_scaler_list(data_list, data_list_for_mean):
    """
    scaled a list of data using mean and standard variation

    :param data_list: list of the session data
    :param data_list_for_mean: the list, which is used for normalisation (trainings data)
    :return: scaled data
    """
    data_for_mean = pd.concat(data_list_for_mean)
    scaled_data = []

    mean_val = np.nanmean(data_for_mean, axis=0)
    # print("mean_val " + str(mean_val))
    std_val = np.nanstd(data_for_mean, axis=0)
    # print("std_val " + str(std_val))

    if std_val[0] == 0:
        std_val[0] = 0.0001
    if std_val[1] == 0:
        std_val[1] = 0.0001

    # Scales dataset
    for d in data_list:
        scaled_data.append((d - mean_val) / std_val)

    return scaled_data


def get_view_size(screen_info, negative_y=True):
    """

    :param screen_info: screen size [width, height] of the user
    :param negative_y: if y should be inverted (used for plotting), since (0,0) is on the top left of a screen
    :return:
    """
    w_size = screen_info['window_xy'].split('x')
    s_size = screen_info['screen_xy'].split('x')
    d_size = screen_info['document_xy'].split('x')

    w_size, s_size, d_size = [transform_to_ints(i) for i in [w_size, s_size, d_size]]
    if negative_y:
        w_size[1], s_size[1], d_size[1] = [size[1] * -1 for size in [w_size, s_size, d_size]]

    return w_size, s_size, d_size


def random_walk(num_points, length_scale, screen, seed=False, plot=False):
    """
    uses Gaussian processes (GPs) to generate a fake trajectory (used for testing)

    :param num_points: number of point for the generated trajectory
    :param length_scale: parameter for the GP for influence the smoothness
    :param screen: screen size in [width, height]
    :param seed: seed to assure reproducibility
    :param plot: if the trajectory should be plotted
    :return: x and y coordinates of the generated trajectory
    """
    X_ = np.linspace(0, 1, num_points)  # start, stop, num points (smoothness)
    kernel = RBF(length_scale=length_scale)  # change "length" of trajectory
    gp = GaussianProcessRegressor(kernel=kernel)
    if seed:
        X = gp.sample_y(X_[:, np.newaxis], 2, random_state=seed)
    else:
        X = gp.sample_y(X_[:, np.newaxis], 2, random_state=np.random.randint(1000))

    trans_X_1 = interp1d([X[:, 0].min(), X[:, 0].max()], [0, screen[0]])
    trans_X_2 = interp1d([X[:, 1].min(), X[:, 1].max()], [0, screen[1]])
    X[:, 0] = trans_X_1([X[:, 0]])
    X[:, 1] = trans_X_2([X[:, 1]])
    if plot:
        fig, ax = plt.subplots()
        param_text = "num_points: " + str(num_points) + "    length_scale: " + str(length_scale)
        ax.plot(X[:, 0], X[:, 1], 'k-')
        # plt.subplots_adjust(right=0.75, bottom=0.18)
        ax.set_title(param_text)
        plt.show()
    return X


def logger_start(logger, args, m_name="--"):
    """
    initialise a logger for training the neural network
    :param logger: the logger
    :param args: arguments given by ArgumentParser
    :param m_name: name of the model (e.g. 1DCNN)
    :return:
    """
    pathlib.Path('../export/logger/').mkdir(parents=True, exist_ok=True)
    hdlr = logging.FileHandler('../export/logger/' + 'logging_' + str(
        datetime.now(tz=None)).replace(' ', '').replace(':', '') + '.log')
    formatter = logging.Formatter('%(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    logger.info("---------------------")
    logger.info(args)
    logger.info(m_name)
    logger.info("---------------------")


def logger_nonseq(name, level=1):
    """
    logger for the non-sequential models
    :param name: name for saving
    :param level: where to save
    :return:
    """
    if level == 2:
        logging.basicConfig(filename='../../export/logger/' + name + ".log",
                            format='%(asctime)s %(message)s',
                            level=logging.INFO)
    if level == 1:
        logging.basicConfig(filename='../export/logger/' + name + ".log",
                            format='%(asctime)s %(message)s',
                            level=logging.INFO)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)
    return logger


def generate_path_ns(path):
    """
    make subfolders
    :param path: path, where to create the subfolders
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(path + '/graphs')
        os.mkdir(path + '/models')
        os.mkdir(path + '/results')


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MetricMeter:
    """
    Computes and stores simple statistics of some metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.min = float('inf')
        self.max = -self.min
        self.last_max = 0
        self.last_min = 0
        self.current = None
        self.mean = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if val > self.max:
            self.max = val
            self.last_max = 0
        else:
            self.last_max += 1
        if val < self.min:
            self.min = val
            self.last_min = 0
        else:
            self.last_min += 1
        self.current = val
        self.sum += val
        self.count += 1
        self.mean = self.sum / self.count


class MeterCollection:
    """
    Collects metrics while training the neural network
    """
    def __init__(self, *names):
        for name in names:
            if name.startswith('_') or name in ('meters', 'update', 'reset'):
                raise ValueError(f'Invalid name `{name}`')
        self.meters = {name: MetricMeter() for name in names}

    def update(self, **kwargs):
        for name, value in kwargs.items():
            self.meters[name].update(value)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def __getattr__(self, name):
        if name in self.meters:
            return self.meters[name]
        else:
            return getattr(super(), name)

    def __repr__(self):
        s = ['{name}={value:.4f}'.format(name=name, value=meter.mean)
             for name, meter in self.meters.items()]
        s = ' '.join(s)
        return s


def get_natural_cubic_spline_model(x, y, minval=None, maxval=None, n_knots=None, knots=None):
    """
    Source: https://github.com/lanadescheemaeker/logistic_models/blob/master/smooth_spline.py
    Get a natural cubic spline model for the data.

    For the knots, give (a) `knots` (as an array) or (b) minval, maxval and n_knots.

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.

    Parameters
    ----------
    x: np.array of float
        The input data
    y: np.array of float
        The outpur data
    minval: float
        Minimum of interval containing the knots.
    maxval: float
        Maximum of the interval containing the knots.
    n_knots: positive integer
        The number of knots to create.
    knots: array or list of floats
        The knots.

    Returns
    --------
    model: a model object
        The returned model will have the following method:
        - predict(x):
            x is a numpy array. This will return the predicted y-values.
    """

    if knots:
        spline = NaturalCubicSpline(knots=knots)
    else:
        spline = NaturalCubicSpline(max=maxval, min=minval, n_knots=n_knots)

    p = Pipeline([
        ('nat_cubic', spline),
        ('regression', LinearRegression(fit_intercept=True))
    ])

    p.fit(x, y)

    return p


class AbstractSpline(BaseEstimator, TransformerMixin):
    """
    Source: https://github.com/lanadescheemaeker/logistic_models/blob/master/smooth_spline.py
    Base class for all spline basis expansions.
    """

    def __init__(self, max=None, min=None, n_knots=None, n_params=None, knots=None):
        if knots is None:
            if not n_knots:
                n_knots = self._compute_n_knots(n_params)
            knots = np.linspace(min, max, num=(n_knots + 2))[1:-1]
            max, min = np.max(knots), np.min(knots)
        self.knots = np.asarray(knots)

    @property
    def n_knots(self):
        return len(self.knots)

    def fit(self, *args, **kwargs):
        return self


class NaturalCubicSpline(AbstractSpline):
    """
    Source: https://github.com/lanadescheemaeker/logistic_models/blob/master/smooth_spline.py
    Apply a natural cubic basis expansion to an array.
    The features created with this basis expansion can be used to fit a
    piecewise cubic function under the constraint that the fitted curve is
    linear *outside* the range of the knots.  The fitted curve is continuously
    differentiable to the second order at all of the knots.
    This transformer can be created in two ways:
      - By specifying the maximum, minimum, and number of knots.
      - By specifying the cutpoints directly.

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.
    Parameters
    ----------
    min: float
        Minimum of interval containing the knots.
    max: float
        Maximum of the interval containing the knots.
    n_knots: positive integer
        The number of knots to create.
    knots: array or list of floats
        The knots.
    """

    def _compute_n_knots(self, n_params):
        return n_params

    @property
    def n_params(self):
        return self.n_knots - 1

    def transform(self, X, **transform_params):
        X_spl = self._transform_array(X)
        if isinstance(X, pd.Series):
            col_names = self._make_names(X)
            X_spl = pd.DataFrame(X_spl, columns=col_names, index=X.index)
        return X_spl

    def _make_names(self, X):
        first_name = "{}_spline_linear".format(X.name)
        rest_names = ["{}_spline_{}".format(X.name, idx)
                      for idx in range(self.n_knots - 2)]
        return [first_name] + rest_names

    def _transform_array(self, X, **transform_params):
        X = X.squeeze()
        try:
            X_spl = np.zeros((X.shape[0], self.n_knots - 1))
        except IndexError:  # For arrays with only one element
            X_spl = np.zeros((1, self.n_knots - 1))
        X_spl[:, 0] = X.squeeze()

        def d(knot_idx, x):
            def ppart(t): return np.maximum(0, t)

            def cube(t): return t * t * t

            numerator = (cube(ppart(x - self.knots[knot_idx]))
                         - cube(ppart(x - self.knots[self.n_knots - 1])))
            denominator = self.knots[self.n_knots - 1] - self.knots[knot_idx]
            return numerator / denominator

        for i in range(0, self.n_knots - 2):
            X_spl[:, i + 1] = (d(i, X) - d(self.n_knots - 2, X)).squeeze()
        return X_spl
