import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    def __init__(self, movie_df):
        self.movie_stats = torch.tensor(movie_df.drop(['description', 'IMDB_Rating'], axis=1).to_numpy())
        self.movie_des = movie_df['description']
        self.movie_ratings = movie_df['IMDB_Rating'].astype(np.float32)

    def __len__(self): return self.movie_stats.shape[0]
    def __getitem__(self, idx: int):
        return self.movie_stats[idx], self.movie_des.iloc[idx], self.movie_ratings.iloc[idx]
    

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