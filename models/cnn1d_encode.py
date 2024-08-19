from src.utils.imports_nn import *
from src.utils import utils as ut_utils


class CNN1D_enc(nn.Module):
    def __init__(self, length, classes=2):
        super(CNN1D_enc, self).__init__()
        self.name = "CNN1D_enc"
        self.length = length
        self.in_channels = 2
        self.kernel_size = 5
        self.padding = 0
        self.stride = 1
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=16, kernel_size=self.kernel_size)
        out_size_1 = int((self.length - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=self.kernel_size)
        out_size_2 = int((out_size_1 - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size)
        out_size_3 = int((out_size_2 - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=self.kernel_size)
        out_size_4 = int((out_size_3 - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=self.kernel_size)
        out_size_5 = int((out_size_4 - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=self.kernel_size)
        out_size_6 = int((out_size_5 - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.fc1 = nn.Linear(128 * out_size_6, 512)
        self.fc2 = nn.Linear(512, classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.conv6(x))
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
    model = CNN1D_enc(length=trajectory_length, classes=2)
    output = model(ft)
    output_class = model.classifier(output)