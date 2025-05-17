from torch import nn

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu6 = nn.ReLU6()
        self.mlp = nn.Sequential(
            nn.Linear(768 + 25, 512), nn.LeakyReLU(.4),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512), nn.LeakyReLU(.4),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512), nn.LeakyReLU(.4),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512), nn.LeakyReLU(.4),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1)
        )

    def forward(self, movie):
      logits = self.mlp(movie)
      return nn.functional.tanh(logits) * 5 + 5