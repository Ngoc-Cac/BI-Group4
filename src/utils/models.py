from torch import nn

from typing import Iterable


class MLPClassifier(nn.Module):
    """
    A Regression using Multi-layer Perceptron architecture for the MovieDataset.
    """
    def __init__(self,
        input_dim: int,
        hidden_dims: Iterable[int] = (512, 512, 512, 512),
        activation_fn: nn.Module | None = None
    ):
        """
        Initialize a Multi-layer Percetron Classifier. By default, the network
        comprises of four hidden layers of dimenion 512 with Leaky ReLU activation
        of negative slope 0.04. Furthermore, between each hidden layer is a batch
        normalization layer for stability.

        The hidden layer dimension may be changed.

        :param int input_dim: The dimension of the input features.
        :param Iterable[int]: A collection of dimensions for each hidden layer.
            This is `(512, 512, 512, 512)` by default.
        :param nn.Module | None: The activation function between each hidden layer.
            This module uses `nn.LeakyReLU(.4)` by default.
        """
        super().__init__()
        if activation_fn is None:
           activation_fn = nn.LeakyReLU(.4)

        self.mlp = nn.Sequential()
        for i, dim in enumerate(hidden_dims):
            self.mlp.add_module(str(i),
                nn.Sequential(
                    nn.Linear(hidden_dims[i - 1] if i != 0 else input_dim, dim),
                    activation_fn,
                    nn.BatchNorm1d(dim)
                )
            )
        self.mlp.add_module("linear_out", nn.Linear(hidden_dims[-1], 1))

    def forward(self, movie):
        """
        Feed the features forward. The output of this network is further
        activated by the Tanh function and scaled to the range [0, 10].

        :param torch.Tensor movie: The features of the movie.
        :return: A tensor of dimension `(batch_size, 1)`.
        :rtype: torch.Tensor
        """
        logits = self.mlp(movie)
        return nn.functional.tanh(logits) * 5 + 5