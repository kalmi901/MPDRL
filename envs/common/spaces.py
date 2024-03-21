from __future__ import absolute_import
from __future__ import annotations

from typing import Union, Optional
from abc import ABC, abstractmethod
import torch
from enum import Enum


class SpaceType(Enum):
    Discrete = 0    # Discrete set of Actions 
    Box = 1         # Continous 
    Hybrid = 2      # Parametrized Action Space


class VSpace(ABC):
    def __init__(self,
                 type: SpaceType,
                 num_envs: int = 1,
                 seed: int = None
                 ) -> None:
        
        self.space = type
        self.num_envs = num_envs
        self.seed = seed
        if seed is not None:
            torch.manual_seed(self.seed)

        self._shape = None
        self._dtype = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"


    @abstractmethod
    def sample(self) -> torch.Tensor:
        raise NotImplementedError
    
    def zeros(self, device: Optional[str] = None) -> torch.Tensor:
        return torch.zeros(size=self._shape,
                           device=self.device if device == None else device)
    
    def ones(self, device: Optional[str] = None) -> torch.Tensor:
        return torch.ones(size=self._shape,
                           device=self.device if device == None else device)
    
    @property
    def shape(self) -> tuple[int]:
        return self._shape
    
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
    
    @property
    def device(self) -> str:
        return self._device
    
    @device.setter
    def device(self, value) -> None:
        self._device = value


class Discrete(VSpace):
    def __init__(self,
                n: int,
                num_envs: int = 1,
                seed: int = None,
                start: int = 0
                ) -> None:
        super().__init__(SpaceType.Discrete, num_envs=num_envs, seed=seed)

        self.n = n
        self.start = start
        self._shape = (self.num_envs, 1)
        self._dtype = torch.long

    def sample(self, device: Optional[str] = None) -> torch.Tensor:
        return torch.randint(low=self.start,
                            high=self.n,
                            size=self._shape,
                            device=self.device if device == None else device)
    

class Box(VSpace):
    def __init__(self, 
                 low: Union[float, torch.Tensor[torch.float]],
                 high: Union[float, torch.Tensor[torch.float]],
                 size: int = 1,
                 num_envs: int = 1,
                 dtype : Union[torch.float32, torch.float64] = None,
                 seed: int  = None
                 ) -> None:
        super().__init__(SpaceType.Box, num_envs=num_envs, seed=seed)

        self._dtype = torch.float32 if dtype == None else dtype
        self._size = size
        if isinstance(low, torch.Tensor):
            self.low = low.to(dtype=self._dtype)
            if size != len(low):
                self._size = len(self.low)
        else:
            if isinstance(low, (int, float)):
                self.low = torch.full((size ,) , low, dtype=self._dtype)

        if isinstance(high, torch.Tensor):
            self.high = high.to(dtype=self._dtype)
            if size != len(high):
                self._size = len(self.high)
        else:
            if isinstance(high, (int, float)):
                self.high = torch.full((size ,) , high, dtype=self._dtype)

        assert self.high.shape == self.low.shape, "Err: observation dimension is not correct!"

        self._shape = (self.num_envs, self._size)
     
    def sample(self, device: Optional[str] = None) -> torch.Tensor:
        device = self.device if device == None else device
        return torch.rand(size=self.shape,
                        dtype=self._dtype,
                        device=device) \
                * (self.high - self.low).to(device) + self.low.to(device)
        


class Hybrid(VSpace):
    def __init__(self, type: SpaceType, num_envs: int = 1, seed: int = None) -> None:
        super().__init__(type, num_envs, seed)

    # TODO:

    def sample(self) -> torch.Tensor:
        pass


if __name__ == "__main__":
    b = Box(1, 3, 3, 5, torch.float64, 2)

    print(b.sample())