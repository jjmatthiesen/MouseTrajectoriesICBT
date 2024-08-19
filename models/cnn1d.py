from src.utils.imports_nn import *
from src.utils import utils as ut_utils


class CNN1D(nn.Module):
    def __init__(self, length, classes=2):
        super(CNN1D, self).__init__()
        self.name = "CNN1D"
        self.length = length
        self.in_channels = 2
        self.kernel_size = 3
        self.padding = 0
        self.stride = 1
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3)
        out_size_1 = int((self.length - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        out_size_2 = int((out_size_1 - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        out_size_3 = int((out_size_2 - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.fc1 = nn.Linear(256 * out_size_3, 512)
        self.fc2 = nn.Linear(512, classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # 1920 x 1200
    trajectory_length = 100
    fake_trajectory = ut_utils.random_walk(num_points=trajectory_length, length_scale=0.06, screen=[1920, 1200], seed=42, plot=True)
    fake_trajectory = torch.tensor(fake_trajectory)

    x = torch.tensor(fake_trajectory[:, 0], dtype=torch.float32)
    y = torch.tensor(fake_trajectory[:, 1], dtype=torch.float32)
    ft = torch.stack((x, y))
    ft = torch.reshape(ft, (1, 2, trajectory_length))
    model = CNN1D(length=trajectory_length, classes=2)
    output = model(ft)

