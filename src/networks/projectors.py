from abc import ABC, abstractmethod
import torch.nn as nn


class Projector(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def reset_weights(self):
        pass

    @abstractmethod
    def get_grad_norm(self):
        pass

    @abstractmethod
    def get_weights_norm(self):
        pass


class FullProjector(Projector):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

    def reset_weights(self):
        nn.init.xavier_normal_(self.layers[0].weight)
        # nn.init.orthogonal_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)

    def get_grad_norm(self):
        g_norm = sum(
            p.grad.norm(2) for p in self.layers[0].parameters() if p.requires_grad
        )
        return g_norm

    def get_weights_norm(self):
        return self.layers[0].weight.norm()


class MLPProjector(Projector):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.layers = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.BatchNorm1d(4*dim),
            nn.ReLU(),
            nn.Linear(4*dim, dim),
        )

    def forward(self, x):
        return self.layers(x)

    def reset_weights(self):
        nn.init.xavier_normal_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)

        nn.init.ones_(self.layers[1].weight)
        nn.init.zeros_(self.layers[1].bias)

        nn.init.xavier_normal_(self.layers[3].weight)
        nn.init.zeros_(self.layers[3].bias)

    def get_grad_norm(self):
        g_norm = sum(
            p.grad.norm(2) for p in self.layers.parameters() if p.requires_grad
        )
        return g_norm

    def get_weights_norm(self):
        return self.layers[0].weight.norm() + self.layers[3].weight.norm()


class LowRankProjector(Projector):
    def __init__(self, dim: int, rank: int = 16):
        super().__init__()
        self.dim = dim
        self.rank = rank

        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.activation = nn.ReLU()

    def forward(self, x):
        y = self.up(self.down(x))
        y = self.activation(y)
        return y

    def reset_weights(self):
        nn.init.xavier_normal_(self.down.weight)
        nn.init.xavier_normal_(self.up.weight)

    def get_grad_norm(self):
        g_norm_down = sum(
            p.grad.norm(2) for p in self.down.parameters() if p.requires_grad
        )
        g_norm_up = sum(p.grad.norm(2) for p in self.up.parameters() if p.requires_grad)
        return g_norm_down + g_norm_up

    def get_weights_norm(self):
        return self.down.weight.norm() + self.up.weight.norm()


class DifferentialProjector(Projector):
    def __init__(self, dim: int, rank: int = 16):
        super().__init__()
        self.dim = dim
        self.part1 = nn.Sequential(
            nn.Linear(dim, rank, bias=False),
            nn.Linear(rank, dim, bias=False),
            nn.ReLU(),
        )
        self.part2 = nn.Sequential(
            nn.Linear(dim, rank, bias=False),
            nn.Linear(rank, dim, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        y_1 = self.part1(x)
        y_2 = self.part2(x)
        y = y_1 - y_2
        return y

    def reset_weights(self):
        nn.init.xavier_normal_(self.part2[0].weight)
        nn.init.xavier_normal_(self.part2[1].weight)

    def get_grad_norm(self):
        g_norm_1 = sum(
            p.grad.norm(2) for p in self.part1.parameters() if p.requires_grad
        )
        g_norm_2 = sum(
            p.grad.norm(2) for p in self.part2.parameters() if p.requires_grad
        )
        return g_norm_1 + g_norm_2

    def get_weights_norm(self):
        return (
            self.part1[0].weight.norm()
            + self.part1[1].weight.norm()
            + self.part2[0].weight.norm()
            + self.part2[1].weight.norm()
        )


class DoubleProjector(Projector):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.part1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.part2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        y1 = self.part1(x)
        y2 = self.part2(x)
        return y1, y2

    def reset_weights(self):
        nn.init.xavier_normal_(self.part2[0].weight)
        nn.init.zeros_(self.part2[0].bias)

    def get_grad_norm(self):
        g_norm1 = sum(
            p.grad.norm(2) for p in self.part1.parameters() if p.requires_grad and p.grad is not None
        )
        g_norm2 = sum(
            p.grad.norm(2) for p in self.part2.parameters() if p.requires_grad and p.grad is not None
        )
        return g_norm1 + g_norm2

    def get_weights_norm(self):
        return self.part1[0].weight.norm() + self.part2[0].weight.norm()
