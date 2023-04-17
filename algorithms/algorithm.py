from abc import ABC, abstractmethod
from numpy.typing import NDArray
from particle import Particle


class Algorithm(ABC):
    @abstractmethod
    def calculate_accelerations(self, particles: list[Particle]) -> \
            NDArray[complex]:
        return NDArray[complex]

    @abstractmethod
    def animate(self, fig, ax, scatter) -> None:
        return


