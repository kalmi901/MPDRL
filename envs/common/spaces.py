from __future__ import absolute_import
from __future__ import annotations

from typing import Union, Generic, TypeVar
from abc import ABC, abstractmethod
import torch
import numpy as np
from enum import Enum


T = TypeVar("T", covariant=True)


class SpaceType(Enum):
    Discrete = 0    # Discrete set of Actions 
    Box = 1         # Continous 
    Hybrid = 2      # Parametrized Action Space


class Space(ABC, Generic[T]):
    def __init__(self,
                 type: SpaceType,
                 **kwargs) -> None:
        self.type = type
        if hasattr(kwargs,"seed"):
            torch.manual_seed(kwargs["seed"])

    @abstractmethod
    def sample(self) -> torch.Tensor[T]:
        raise NotImplementedError



class Discrete(Space[T]):
    def __init__(self,
                n: int | np.int64,
                seed: int | np.int64
                ) -> None:
        super().__init__(Discrete)

        self.n = np.float64(n)
        self.seed = np.float64(seed)


    def sample(self, mask = None) -> torch.Tensor[T]:
        print("print mee")
        


if __name__ == "__main__":
    d = Discrete(3, 42)

    d.sample()