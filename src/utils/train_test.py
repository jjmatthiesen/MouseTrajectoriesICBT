from src.utils.imports import *
from src.utils.imports_nn import *
from src.utils import utils as ut_utils
import src.utils.plots as ut_plots


def get_batch_data(data, index_list):
    """
    gets a batch of data based n indexes
    :param data: list of sessions
    :param index_list: the indexes of the sessions
    :return: a batch of data
    """
    batch_data = []
    for i in index_list:
        batch_data.append(data[i].to(dtype=torch.float32))

    return batch_data


def get_blocks(args, data, win_size, step_size=np.nan):
    """
    Create block from a whole trajectory by cutting it into smaller sub-trajectories
    :param args: arguments given by ArgumentParser
    :param data:  The trajectory data
    :param win_size: size on the window
    :param step_size: if step_ze = 1, step_size = win_size = traj-length
    :return: blocks of the trajectory
    """
    if np.isnan(step_size):
        # is no step_size is given, block will not overlap
        return torch.split(data, args.trajectory_length)
    blocks = []
    for i in range(0, len(data), int(step_size)):
        if i + win_size <= len(data):
            blocks.append(data[i:i + win_size])
    return blocks


def save_result_as_csv(m, args, fo, scope, train_acc, train_loss, t_acc, t_loss, t_auc):
    """
    :param m: the model
    :param args: arguments given by ArgumentParser
    :param fo: the fold number
    :param scope: if train_val or train_test
    :param train_acc: the train accuracy
    :param train_loss: the train loss
    :param t_acc: the test accuracy
    :param t_loss: the test loss
    :param t_auc:  the test auc
    :return:
    """
    pathlib.Path('../results/' + scope).mkdir(parents=True, exist_ok=True)
    with open('../results/' + scope + '/results_' + m.name + '.csv', 'a', encoding='utf-8') as f:
        f.write(
            f'{"holdout_strat"}, '
            f'{"lr_"}{args.lr}, '
            f'{"bs_"}{args.bs}, '
            f'{"ep_"}{args.epochs}, '
            f'{"tj-length_"}{args.trajectory_length}, '
            f'{"seed_"}{args.seed},'
            f'{"fold_"}{fo},'
            f'{"step_size_"}{args.step_size},'
            f'{"pauses_"}{args.pauses},'
            f'{"_"}{args.extra},'
            f' {train_acc}, {train_loss}, {t_acc}, {t_loss}, {t_auc} \n')


def train_model(m, m_save_name, s_loader_train, train_data, s_loader_val, val_data, o_fo, fo, dev, crit, opt, args):
    """
    trains the neural network
    :param m: the model
    :param m_save_name: the name to save results/ logs etc.
    :param s_loader_train: the sample data loader for training
    :param train_data: the trainings data
    :param s_loader_val: the sample data loader for validation
    :param val_data: the validation data
    :param o_fo: the number of the outer fold
    :param fo: the number of the inner fold
    :param dev: the device (cpu or gpu)
    :param crit: the criterion/ loss calculation
    :param opt: the optimizer
    :param args: arguments given by ArgumentParser
    :return:
    """
    logger = logging.getLogger('logging_performance')
    if args.log:
        ut_utils.logger_start(logger, args, m.name)
    train_acc, train_losses = [], []
    val_accuracies, val_losses = [], []
    epoch_accuracy, epoch_loss, val_accuracy, val_loss, val_auc = "-", "-", "-", "-", "-"
    labels_val, pre_probs_val = [], []
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_auc = 0
        meters = ut_utils.MeterCollection("loss", "acc")
        for indexes, labels in s_loader_train:
            data_batch = get_batch_data(train_data.data, indexes)
            m.train()
            predictions = []
            pre_probabilities = []
            loss = 0
            for idx, d in enumerate(data_batch):
                outcome_model = []
                if len(d) < args.trajectory_length:
                    # padding with 0
                    d = F.pad(input=d, pad=(0, 0, 0, args.trajectory_length - len(d)), mode='constant', value=0)
                # split data in blocks
                if args.step_size != 1.0:
                    blocks = get_blocks(args, d, args.trajectory_length,
                                        step_size=int(args.trajectory_length * args.step_size))
                else:
                    blocks = get_blocks(args, d, args.trajectory_length)
                for b in blocks:
                    b = b.T.unsqueeze(0)
                    out = m(b)
                    outcome_model.append(out)
                cla = m.classifier(torch.cat(outcome_model))
                pre = cla[:, 1].mean()
                prediction = 1 if pre > 0.5 else 0
                pre_probabilities.append(pre.item())
                predictions.append(prediction)
                loss += crit(torch.mean(cla, dim=0), labels[idx])

            # calculate loss after batch
            # loss = criterion(predictions, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            acc = ((torch.tensor(predictions) == labels).float().mean()).item()
            auc_train = roc_auc_score(labels, pre_probabilities)
            meters.update(loss=loss.item(), acc=acc)
            epoch_accuracy += acc / len(s_loader_train)
            epoch_loss += loss.item() / len(s_loader_train)
            epoch_auc += auc_train / len(s_loader_train)

        print('outer fold : {}, fold : {}, epoch : {}, train accuracy : {}, train loss : {}'.format(o_fo, fo, epoch + 1, epoch_accuracy, epoch_loss))
        if args.log:
            logger.info("epoch={:4d} {}".format(epoch, meters))
        train_acc.append(epoch_accuracy)
        train_losses.append(epoch_loss)

        val_accuracy, val_loss, val_auc, labels_val, pre_probs_val = test_model(m, s_loader_val, val_data, dev, crit, args)

        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
    if args.plot_performance:
        ut_plots.plot_performance(train_acc, val_accuracies, m.name + "/" + args.scope, m_save_name,
                                  labels=["train acc", "val acc"],
                                  what="accuracy", save=True)
        ut_plots.plot_performance(train_losses, val_losses, m.name + "/" + args.scope, m_save_name,
                                  labels=["train loss", "val loss"],
                                  what="loss", save=False)
        plt.close()
    if args.log:
        logging_save_path = '../export/logging/' + m.name + '/' + args.scope + '/holdout_strat/'
        pathlib.Path(logging_save_path).mkdir(parents=True, exist_ok=True)
        df_logging = pd.DataFrame(
            data={'train_acc': train_acc,
                  'train_loss': train_losses,
                  'val_acc': val_accuracies,
                  'val_losses': val_losses}
        )
        df_logging.to_csv(logging_save_path + m_save_name + '_' + args.scope + '.csv',
                          index_label='epoch')

    save_result_as_csv(m, args, fo, args.scope, epoch_accuracy, epoch_loss, val_accuracy, val_loss, val_auc)
    # export model
    pathlib.Path('../export/models/').mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), '../export/models/' + m_save_name + '.pt')

    return epoch_accuracy, epoch_loss, val_auc, labels_val, pre_probs_val


def test_model(m, s_loader, test_data, dev, crit, args):
    """

    :param m: the model
    :param s_loader: the sample data loader for testing
    :param test_data: the test data
    :param dev: the device (cpu or gpu)
    :param crit: the criterion/ loss calculation
    :param args: arguments given by ArgumentParser
    :return:
    """
    m.eval()
    t_accuracy = 0
    t_loss = 0
    labels_test_all = []
    pre_probs_all = []
    # ======== Test =====
    for i, labels_test in s_loader:
        data = get_batch_data(test_data.data, i)
        labels_test = labels_test.type(torch.LongTensor)
        labels_test = labels_test.to(dev)
        preds = []
        pre_probs = []
        classifications = []
        for j, d in enumerate(data):
            outcome_model = []
            if len(d) < args.trajectory_length:
                # padding with 0
                d = F.pad(input=d, pad=(0, 0, 0, args.trajectory_length-len(d)), mode='constant', value=0)
            # split data in blocks
            if args.step_size != 1:
                blocks = get_blocks(args, d, args.trajectory_length, step_size=int(args.trajectory_length * args.step_size))
            else:
                blocks = get_blocks(args, d, args.trajectory_length)
            for b in blocks:
                b = b.T.unsqueeze(0)
                out = m(b)
                outcome_model.append(out)
                # predictions_user.append(int(torch.argmax(cla, dim=1)))
            cla = m.classifier(torch.cat(outcome_model))
            pre = cla[:, 1].mean()
            prediction = 1 if pre > 0.5 else 0
            preds.append(prediction)
            pre_probs.append(pre.item())
            classifications.append(cla)
            t_loss += crit(cla, torch.tensor(len(cla) * [labels_test[j]]))

        acc = ((torch.tensor(preds) == labels_test).float().mean()).item()
        # auc_test = roc_auc_score(labels_test, pre_probs)
        t_accuracy += acc / len(s_loader)
        t_loss += t_loss.item() / len(s_loader)
        # t_auc += auc_test / len(s_loader)
        labels_test_all.extend(labels_test.tolist())
        pre_probs_all.extend(pre_probs)
    t_auc = roc_auc_score(labels_test_all, pre_probs_all)
    if args.scope == "train_val":
        print('val_accuracy : {}, val_loss : {}'.format(t_accuracy, t_loss))
    else:
        print('test_accuracy : {}, test_loss : {}'.format(t_accuracy, t_loss))
    return t_accuracy, t_loss.item(), t_auc, labels_test_all, pre_probs_all