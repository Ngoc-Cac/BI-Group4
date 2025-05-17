from torch import nn

class MLPClassifier(nn.Module):
    def __init__(self,
        input_dim: int,
        hidden_dims: tuple[int] = (512, 512, 512, 512),
        activation_fn: nn.Module | None = None
    ):
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
      logits = self.mlp(movie)
      return nn.functional.tanh(logits) * 5 + 5