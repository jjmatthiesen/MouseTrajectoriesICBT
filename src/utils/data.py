from src.utils.imports_nn import *
from src.utils.imports import *
from src.utils import utils as ut_utils


class DatasetMaker(Dataset):
    """
    created a Dataset encompassing data and labels
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        class_label = self.labels[i]
        trajectory = self.data[i]
        return trajectory, class_label


def get_trajectory_set(length_scale=0.06, n=100, tra_length=50):
    """
    create an artificial dataset of trajectories created with Gaussian processes (GPs)
    :param length_scale: parameter for the GP for influence the smoothness
    :param n: amount of trajectories, which should be generated
    :param tra_length: number of point for the generated trajectory
    :return:
    """
    trajectory_set = []
    gen_seeds = random.sample(range(n * 10), n)
    print(gen_seeds[0])
    for i in gen_seeds:
        fake_trajectory = ut_utils.random_walk(num_points=tra_length, length_scale=length_scale,
                                               screen=[1920, 1200], seed=i,
                                               plot=False)
        x = torch.tensor(fake_trajectory[:, 0], dtype=torch.float32)
        y = torch.tensor(fake_trajectory[:, 1], dtype=torch.float32)
        ft = torch.stack((x, y))
        trajectory_set.append(ft)
    return torch.stack(trajectory_set, dim=0)


def get_synthetic_train_data(length_scales, n=1000, t_length=50):
    """
    works just for two classes
    :param t_length: length od the generated trajectories
    :param length_scales: two variables for two different sets of trajectories
    :param n: how many trajectories
    :return: concatenated data and labels
    """
    sets_data = []
    sets_labels = []
    for i, ls in enumerate(length_scales):
        trajectory_set = get_trajectory_set(length_scale=ls, n=n, tra_length=t_length)
        # plot generated trajectory
        # plot_trajectory(trajectory_set_006[0], "0.06")
        # plot_trajectory(trajectory_set_002[0], "0.02")
        if i == 0:
            trajectory_set_y = torch.zeros(len(trajectory_set), dtype=torch.float32)
        else:
            trajectory_set_y = torch.ones(len(trajectory_set), dtype=torch.float32)
        sets_data.append(trajectory_set)
        sets_labels.append(trajectory_set_y)
    trajectory_data = torch.cat((sets_data[0], sets_data[1]))
    class_labels = torch.cat((sets_labels[0], sets_labels[1]))
    return trajectory_data, class_labels
