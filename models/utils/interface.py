from abc import ABC, abstractmethod

from utils.c2n_map import C2NMap


class DecipherBase(ABC):
    def __init__(self, c2n_map: C2NMap):
        self.c2n_map = c2n_map

    @abstractmethod
    def perform(self, text: str) -> str:
        raise NotImplementedError
