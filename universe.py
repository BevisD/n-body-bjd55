from algorithms import Algorithm
from particle import Particle
from integration import euler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy.typing import NDArray


class Universe:
    def __init__(self, particles: list[Particle], algorithm: Algorithm,
                 dt: float, integration: callable = euler):
        self.algorithm = algorithm
        self.particles = particles
        self.integration = integration
        self.dt = dt
        self.T = 0

    def update(self) -> None:
        self.integration(self.particles, self.algorithm, self.dt)
        self.T += self.dt

    def get_particle_coordinates(self) -> tuple[
        NDArray[float], NDArray[float]]:
        s = [(p.centre.real, p.centre.imag) for p in self.particles]
        x, y = np.transpose(s)
        return x, y

    def animation(self, frames: int = 100) -> None:
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        x_0, y_0 = self.get_particle_coordinates()
        scatter = ax.scatter(x_0, y_0, s=1)

        def animate(i) -> None:
            self.update()
            coordinates = self.get_particle_coordinates()
            scatter.set_offsets(np.transpose(coordinates))
            self.algorithm.animate(fig, ax, scatter)

        anim = animation.FuncAnimation(fig, animate, frames, interval=30)
        plt.show()
