import torch


class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.extractor = torch.nn.Sequential(
            torch.nn.ZeroPad2d(2),
            torch.nn.Conv2d(1, 6, 5),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(16, 120, 5),
            torch.nn.Tanh(),
        )

        self.classfier = torch.nn.Sequential(
            torch.nn.Linear(120, 84),
            torch.nn.Tanh(),
            torch.nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        x = self.classfier(x)

        return x
