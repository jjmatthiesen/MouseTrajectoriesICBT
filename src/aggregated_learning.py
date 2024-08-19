from src.utils import data as ut_data
from models.cnn1d_antal_feher import *
from models.cnn1d_encode import *
from src.utils.imports import *
import src.utils.train_test as ut_train


def parse_args():
    """
    :argument epochs: the amount of epochs for training
    :argument lr: the learning rate
    :argument trajectory-lengths: the length of the sub-trajectories
    :argument seeds: seed for multiple runs
    :argument plot_performance: if the performance should be plotted
    :argument log: if the training history should be saved
    :argument step_size: step size defines the overlap. e.g 0.5 will lead to half of the trajectory length as overlap
    :argument pauses: if pauses (no movement of the cursor) should be taken into consideration
    :argument normalize_screen: Coordinates are normalised using std and variance.
    If this is set to true, they are additionally normalised in relation to the screen size.
    :argument scope: "train_val" when running just train and validation runs,
    this toggles the plotting of validation results.
    :argument test: If the model should be tested. Scope should be "train_test" in that case
    :argument outcome_file: the file for the outcome labels
    :argument extra: additional information, for preliminary testing

    :return: the set hyperparameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--trajectory-lengths', nargs='+', default=[100, 128, 256])
    parser.add_argument('--seeds', nargs='+', default=[1, 11, 111, 42, 66])
    parser.add_argument('--plot_performance', type=bool, default=True)
    parser.add_argument('--log', type=bool, default=True)
    # step size defines the overlap. e.g 0.5 will lead to half of the trajectory length as overlap
    parser.add_argument('--step_size', type=bool, default=0.5)
    parser.add_argument('--pauses', type=bool, default=False)
    parser.add_argument('--normalize_screen', type=bool, default=False)
    parser.add_argument('--scope', type=str, default="train_val")
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--outcome_file', type=str, default="20240703_outcomes.csv")
    parser.add_argument('--extra', type=str, default="")
    return parser.parse_args()


def pre_process_data(path, args, test_set_ids, train_set_ids):
    """
    We load all the data from each user, concat the sessions and cut the length,
    so that it is a multiple of the trajectory length
    :param train_set_ids: IDs of the participant in training
    :param test_set_ids: IDs of the participant in testing
    :param path: path for the data
    :param args: arguments
    :return: data and labels
    """
    folders = glob(path)
    dataset_train = []
    dataset_test = []
    class_labels_train = []
    class_labels_test = []

    # load screen info
    path_screen_users = "../data/mouse_data/raw/mousetracker.csv"
    df_users_screen = pd.read_csv(path_screen_users)

    # dropout and non-dropout
    for i, folder in enumerate(folders):
        class_dataset_train = []
        class_dataset_test = []
        print(folder)
        for user in glob(folder + "/pre/*"):
            user_dataset = pd.DataFrame()
            for file in glob(user + "/*.csv"):
                file_data = pd.read_csv(file, usecols=['client_x', 'client_y', 'eventtime', 'eventname'])[
                    ['client_x', 'client_y', 'eventtime', 'eventname']]
                if args.normalize_screen:
                    # normalize using screen info
                    rec_id = file.split("id")[-1].split("__")[0]
                    user_screen_info = df_users_screen.loc[(df_users_screen['id'] == int(rec_id))]
                    s_size = user_screen_info.iloc[0]['screen_xy'].split('x')
                    file_data['client_x'] = file_data['client_x'].div(int(s_size[0]))
                    file_data['client_y'] = file_data['client_y'].div(int(s_size[1]))

                user_dataset = pd.concat(
                    ([user_dataset, file_data[['client_x', 'client_y', 'eventtime', 'eventname']]]))

            # print("Length of files from " + user + ": " + str(len(user_dataset)))
            user_dataset.sort_values('eventtime', inplace=True)
            user_dataset = user_dataset.reset_index(drop=True)
            user_dataset['dt'] = user_dataset['eventtime'].diff()
            if not args.pauses:
                user_dataset = user_dataset.loc[~user_dataset['eventname'].isin(['pause'])]
            user_dataset = calculate_diff(user_dataset[['client_x', 'client_y', 'eventtime']], coords="client")
            # get velocity
            user_dataset['vx'] = user_dataset['dx'] / user_dataset['dt']
            user_dataset['vy'] = user_dataset['dy'] / user_dataset['dt']
            user_df_velocity = pd.concat([user_dataset['vx'], user_dataset['vy']], axis=1)
            # delete inf and nan
            user_df_velocity = user_df_velocity[
                user_df_velocity.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
            # # max length, which is a multiple of the trajectory length
            # max_length = user_df_velocity.shape[0] - (user_df_velocity.shape[0] % args.trajectory_length)

            if int(user.split('/')[-1]) in train_set_ids:
                if len(user_df_velocity) > 0:
                    class_dataset_train.append(user_df_velocity)
                    class_labels_train.append(torch.tensor([i]))
            elif int(user.split('/')[-1]) in test_set_ids:
                if len(user_df_velocity) > 0:
                    class_dataset_test.append(user_df_velocity)
                    class_labels_test.append(torch.tensor([i]))
            else:
                print("user not in outcomes")

        # print('Dataset shape from folder: ' + folder + ' - ' + str(class_dataset.shape))
        for cd in class_dataset_train:
            dataset_train.append(cd)
        for cd in class_dataset_test:
            dataset_test.append(cd)

    return dataset_train, torch.cat(class_labels_train), dataset_test, torch.cat(class_labels_test)


def get_model_save_name(m_name, args, o_fo, fo):
    """
    :param m_name: the name of the model (e.g. 1DCNN)
    :param args: arguments given by ArgumentParser
    :param o_fo: number older fold
    :param fo: number inner fold
    :return: a name with all hyperparameter information
    """
    m_saved_name = str(m_name) + \
                   "_holdout_strat" + \
                   "_lr_" + str(args.lr) + \
                   "_bs_" + str(args.bs) + \
                   "_ep_" + str(args.epochs) + \
                   "_tj-length_" + str(args.trajectory_length) + \
                   "_seed_" + str(args.seed) + \
                   "_outer-fold_" + str(o_fo) + \
                   "_inner-fold_" + str(fo) + \
                   "_ss_" + str(args.step_size) + \
                   "_p_" + str(args.pauses)
    if args.extra != "":
        m_saved_name += "_ex_" + str(args.extra)
    return m_saved_name


def calculate_diff(df, coords="client"):
    """
    :param df: a dataframe encompassing time, x, and y coordinates
    :param coords: which coordinates to take. "client" or "page".
    These are the two different coordinated the tracker records.

    :return: the derivative of the columns
    """
    # Calculates the difference between two consecutive row
    df = df.diff()

    # The first row values are NaN, because of using diff()
    if coords == "client":
        df = df[1:].rename(columns={'client_x': 'dx', 'client_y': 'dy', 'eventtime': 'dt'})
    else:
        df = df[1:].rename(columns={'page_x': 'dx', 'page_y': 'dy', 'eventtime': 'dt'})
    df = df.reset_index()
    return df


def get_sample_loader(data, args):
    """
    create a dataset of indexes for training_data
    this is used to reference the data while training/validation

    :param data: an object of data and labels
    :param args: arguments given by ArgumentParser
    :return:
    """
    num = [*range(len(data.data))]
    sample_data = ut_data.DatasetMaker(torch.tensor(num), data.labels)
    return DataLoader(sample_data, batch_size=args.bs, shuffle=True)


def set_max_length(data, labels, args):
    """
    cuts all session to the maximum length, which is a multiple of the trajectory_length

    :param args:  arguments given by ArgumentParser
    :param data:  the trajectory data
    :param labels: the labels for training
    :return:
    """
    dataset, labels_ = [], []
    for d, l in zip(data, labels):
        # max length, which is a multiple of the trajectory length
        max_length = d.shape[0] - (d.shape[0] % args.trajectory_length)
        if len(d[:max_length]) > 0:
            dataset.append(d[:max_length])
            labels_.append(l)
    return dataset, labels_


if __name__ == '__main__':
    # ====== args ========
    arguments = parse_args()
    # best performing arguments are determent in inner CV
    best_arguments = copy.deepcopy(arguments)
    best_arguments.trajectory_length = arguments.trajectory_lengths[1]
    mouse_data_train_test = '../data/mouse_data/participants/desktop/train_test/*'
    outcomes_path = "../data/" + arguments.outcome_file

    outcomes = pd.read_csv(outcomes_path)
    outcomes = outcomes[outcomes['pre_mousedata'] == 1].copy(deep=True)

    # 5-fold cross validation
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_plot_data = {}

    for fold_o, (train_o_id, test_ids) in enumerate(cv_outer.split(outcomes['Internt ID'], outcomes['dropout_mod'])):
        print("-------------------")
        print("Outer_Fold: " + str(fold_o))
        print("-------------------")
        train_intern_ids = list(outcomes['Internt ID'].iloc[j] for j in train_o_id)
        test_intern_ids = list(outcomes['Internt ID'].iloc[j] for j in test_ids)
        train_val_data_pp, train_val_labels_pp, test_data_pp, test_labels_pp = pre_process_data(
            mouse_data_train_test, arguments, test_intern_ids, train_intern_ids)
        all_preds = []
        all_labels = []
        auc_plot_data[fold_o] = {}
        for seed in arguments.seeds:
            arguments.seed = seed
            auc_plot_data[fold_o][seed] = {}

            # =========Set seed for training the network===========
            random.seed(arguments.seed)
            np.random.seed(arguments.seed)
            torch.manual_seed(arguments.seed)
            torch.use_deterministic_algorithms(True)
            # ====================

            best_val_auc = 0
            criterion = nn.CrossEntropyLoss()
            dataset_test_scale = ut_utils.standard_scaler_list(test_data_pp, train_val_data_pp)
            dataset_test = ut_utils.transform_to_torch_list(dataset_test_scale)
            test_data = ut_data.DatasetMaker(dataset_test, test_labels_pp)

            for tl in arguments.trajectory_lengths:
                arguments.trajectory_length = tl
                val_aucs = []
                # inner fold for hyper parameter opt
                for fold, (train, val) in enumerate(cv_inner.split(train_val_data_pp, train_val_labels_pp)):
                    data_train = list(train_val_data_pp[j] for j in train)
                    data_val = list(train_val_data_pp[j] for j in val)

                    l_train = train_val_labels_pp[train]
                    l_val = train_val_labels_pp[val]

                    # set max length based on arguments.trajectory_length
                    dataset_train, labels_train = set_max_length(data_train, l_train, arguments)
                    dataset_val, labels_val = set_max_length(data_val, l_val, arguments)

                    # scale data
                    dataset_train_scale = ut_utils.standard_scaler_list(dataset_train, dataset_train)
                    dataset_val_scale = ut_utils.standard_scaler_list(dataset_val, dataset_train)

                    # create list of tensors
                    dataset_train = ut_utils.transform_to_torch_list(dataset_train_scale)
                    dataset_val = ut_utils.transform_to_torch_list(dataset_val_scale)

                    training_data = ut_data.DatasetMaker(dataset_train, labels_train)
                    val_data = ut_data.DatasetMaker(dataset_val, labels_val)

                    # datasets for creating batches based on indexes for training_data
                    # this is used later to reference the data while training/validation
                    sample_loader_train = get_sample_loader(training_data, arguments)
                    sample_loader_val = get_sample_loader(val_data, arguments)

                    # model = CNN1D_enc(length=arguments.trajectory_length, classes=2)
                    model = CNN1D_ANTAL(length=best_arguments.trajectory_length, classes=2)

                    device = 'cpu'
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=arguments.lr)
                    model.to(device)

                    model_saved_name = get_model_save_name(model.name, arguments, fold_o, fold)

                    # train network and validate on val data:
                    train_accuracy, test_loss, val_auc, labels_vali, pre_probabilities_vali = (
                        ut_train.train_model(model,
                                             model_saved_name,
                                             sample_loader_train,
                                             training_data,
                                             sample_loader_val,
                                             val_data,
                                             fold_o,
                                             fold,
                                             device, criterion, optimizer, arguments))

                    val_aucs.append(val_auc)
                if np.mean(val_aucs) > best_val_auc:
                    # update parameter
                    best_val_auc = np.mean(val_aucs)
                    best_arguments.trajectory_length = arguments.trajectory_length

            if arguments.test:
                # test on test data
                best_arguments.seed = seed
                # model_test = CNN1D_enc(length=best_arguments.trajectory_length, classes=2)
                model_test = CNN1D_ANTAL(length=best_arguments.trajectory_length, classes=2)
                model_saved_name = get_model_save_name(model_test.name, best_arguments, fold_o, fo=1)
                # load state dict
                model_test.load_state_dict(torch.load('../export/models/' + model_saved_name + '.pt', map_location='cpu'))
                device = 'cpu'
                sample_loader_test = get_sample_loader(test_data, best_arguments)

                test_accuracy, test_loss, test_auc, labels_test, pre_probabilities_test = (
                    ut_train.test_model(model_test, sample_loader_test, test_data, device, criterion, best_arguments))
                ut_train.save_result_as_csv(model_test, best_arguments, fold_o, "test", "", "", test_accuracy, test_loss, test_auc)

                # concatenate for all seeds
                all_preds.extend(pre_probabilities_test)
                all_labels.extend(labels_test)

                auc_plot_data[fold_o][seed]['predict_probas'] = pre_probabilities_test
                auc_plot_data[fold_o][seed]['y_true'] = labels_test
        if arguments.test:
            pathlib.Path('../results/').mkdir(parents=True, exist_ok=True)
            with open("../results/pred_probabilities_CNN1D_ANTAL.json", "w") as outfile:
                json.dump(auc_plot_data, outfile)


