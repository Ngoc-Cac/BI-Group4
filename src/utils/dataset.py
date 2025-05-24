import numpy as np
import torch

from torch.utils.data import Dataset

class MovieDataset(Dataset):
    """
    Dataset for loading movies data. This object is indexable and return a
        tuple (movie_statistics, movie_description, movie_rating).
    """
    def __init__(self, movie_df):
        self.movie_stats = torch.tensor(movie_df.drop(['description', 'IMDB_Rating'], axis=1).to_numpy())
        self.movie_des = movie_df['description']
        self.movie_ratings = movie_df['IMDB_Rating'].astype(np.float32)

    def __len__(self): return self.movie_stats.shape[0]
    def __getitem__(self, idx: int):
        return self.movie_stats[idx], self.movie_des.iloc[idx], self.movie_ratings.iloc[idx]