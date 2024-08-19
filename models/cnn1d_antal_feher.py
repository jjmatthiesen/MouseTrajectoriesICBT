from src.utils.imports_nn import *
from src.utils import utils as ut_utils


class CNN1D_ANTAL(nn.Module):
    def __init__(self, length, classes=2):
        super(CNN1D_ANTAL, self).__init__()
        self.name = "CNN1D_ANTAL"
        self.length = length
        self.kernel_size_conv11 = 6
        self.kernel_size_conv12 = 3
        self.stride_conv11 = 2
        self.stride_conv12 = 1
        self.padding = 0
        self.kernel_size_conv21 = 4
        self.kernel_size_conv22 = 2
        self.stride_conv21 = 2
        self.stride_conv22 = 1
        # ----- tower 1 ------
        self.conv11 = nn.Conv1d(in_channels=2, out_channels=40, kernel_size=self.kernel_size_conv11, stride=self.stride_conv11)
        # out_size_11 = int((self.length - self.kernel_size_conv11 + 2 * self.padding) / self.stride_conv11) + 1
        self.conv12 = nn.Conv1d(in_channels=40, out_channels=60, kernel_size=self.kernel_size_conv12, stride=self.stride_conv12)
        # out_size_12 = int((out_size_11 - self.kernel_size_conv12 + 2 * self.padding) / self.stride_conv12) + 1

        # ----- tower 2 ------
        self.conv21 = nn.Conv1d(in_channels=2, out_channels=40, kernel_size=self.kernel_size_conv21, stride=self.stride_conv21)
        self.conv22 = nn.Conv1d(in_channels=40, out_channels=60, kernel_size=self.kernel_size_conv22, stride=self.stride_conv22)
        self.maxPool = nn.AdaptiveMaxPool1d(output_size=60)
        self.dropout = nn.Dropout(0.15)
        # each tower has a max pool at the end to reduce the dim to 60
        # output_size from maxPool: 60 * 2 * out_channel (conv)
        self.fc1 = nn.Linear(120*60, 60)
        self.fc2 = nn.Linear(60, classes)

    def forward(self, x_in):
        x1 = nn.functional.relu(self.conv11(x_in))
        x1 = nn.functional.relu(self.conv12(x1))
        x1 = self.maxPool(x1)
        x2 = nn.functional.relu(self.conv21(x_in))
        x2 = nn.functional.relu(self.conv22(x2))
        x2 = self.maxPool(x2)
        x = torch.concat([x1, x2], dim=1)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        return x

    def classifier(self, x):
        x = nn.functional.relu(self.fc1(x))
        m = nn.Softmax(dim=1)
        x = m(self.fc2(x))
        return x


if __name__ == '__main__':
    # 1920 x 1200
    trajectory_length = 128
    fake_trajectory = ut_utils.random_walk(num_points=trajectory_length, length_scale=0.06, screen=[1920, 1200], seed=42, plot=True)
    fake_trajectory = torch.tensor(fake_trajectory)

    x = torch.tensor(fake_trajectory[:, 0], dtype=torch.float32)
    y = torch.tensor(fake_trajectory[:, 1], dtype=torch.float32)
    ft = torch.stack((x, y))
    ft = torch.reshape(ft, (1, 2, trajectory_length))
    model = CNN1D_ANTAL(length=trajectory_length, classes=2)
    output = model(ft)


