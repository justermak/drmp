from abc import ABC, abstractmethod

import torch


class Normalizer(ABC):
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def fit(self, X: torch.Tensor):
        self.X = X
        if X.ndim > 2:
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X

        self.mins = X_flat.min(dim=0).values
        self.maxs = X_flat.max(dim=0).values

        self.constant_mask = (self.maxs - self.mins).abs() < self.eps
        self.range = self.maxs - self.mins
        self.range = torch.where(
            self.constant_mask, torch.ones_like(self.range), self.range
        )

    @abstractmethod
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class LimitsNormalizer(Normalizer):
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mins) / self.range
        x = 2 * x - 1 + self.constant_mask.float()
        return x

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clip(x, -1, 1)
        x = (x + 1 - self.constant_mask.float()) / 2
        x = x * self.range + self.mins
        return x


def get_normalizers():
    return {"LimitsNormalizer": LimitsNormalizer}
