from torch import nn

class LaneFollower(nn.Module):
    def __init__(self):
        super(LaneFollower,self).__init__()
        self.conv_layer = nn.Sequential(
            # 200 75
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 4), padding=0, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            # 98 36
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 5), padding=0, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            # 48 16
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=10*23*7, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200,out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50,out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10,out_features=1)
        )

    def forward(self, input):
        output = self.conv_layer(input)
        output = self.linear_layers(output)
        return output
