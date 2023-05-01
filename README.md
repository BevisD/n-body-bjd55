# Hierarchical methods for $N$-body simulation

**The written report for this project is in the** ```report.ipynb``` **notebook**

This repository contains the functionality to simulate a universe of particles, each inducing a force on the other. Containing different algorithms for the interactions of particles; a variety of numerical integration schemes; and a collection of different force types; the effect that each of these factors has on the performance can be investigated. 

## How to use this repository
To simulate a universe of interacting particles, a few things are needed.

### Particles
There is a ```Particle``` class containing particle specific data, such as charge, mass, position etc...
The positions of the particles are stored as complex numbers with the real/imaginary components corresponding to the x/y real-space coordinates.


### Forces
The ```Force``` class contains two functions and a number of parameters to adjust the strength/range of the force. The functions that the force class contains are ```calculate_force(particle1, particle2)``` and ```calculate_potential(particle1, particle2)``` which calculate the force and potential between two particles.

### Integration
To simulate the particles' motions, the positions must be numerically integrated over time. There is a collection of integration schemes that update the positons and velocities of each particle for a given time-step $dt$. Only the Euler and Runge-Kutta 4 integration schemes have been implemented as a proof of principle, as this is not what the computing project is investigating.

### Algorithm
The ```Algorithm``` class is the object that applies the forces to the particles. The ```PairWise``` algorithm simply iterates through every pair of particles and calculates the forces between them. The ```BarnesHut``` and ```FMM``` algorithms are more sophisticated and have parameters that affect their speed and accuracy.

### Universe
The ```Universe``` class combines all of the previously mentioned components to run the simulation of the particles, produce animations, and calculate metrics like momentum, energy etc...

An example of how to display an animation of the particles motions is given in main.py, however a quick example is displayed below:
```py
from particle import Particle
from algorithms import BarnesHut
from forces import InverseSquare
from universe import Universe

  N = 100 # The number of particles to create
  G = -1 # The scaling constant for the force, negative for gravitational attraction
  SOFTENING = 0.01 # The softening distance for the force
  DT = 0.005 # The numerical integration time-step
  THETA = 0.5 # The Barnes-Hut theta parameter


  # Create the force object
  force = InverseSquare(G, SOFTENING)

  # Create the algorithm with its parameter and force
  BH_algorithm = BarnesHut(force, theta=THETA)

  # Create the particles for our universe
  particles = [
      Particle(charge=1 / np.sqrt(N)) for _ in
      range(N)]

  # Create the universe, attaching the particles and algorithm
  universe = Universe(particles, BH_algorithm, DT)
  universe.animation() # Display the motion of the particles
```

The ```universe.animation``` funciton calls ```universe.update()``` for each frame and displays the positions of the particles to a matplotlib scatter plot. If a filename is given then the animation will be saved to the ```animations``` directory. Additionally, the ```iters_per_frame``` argument is set to $1$ by default, however if changed, ```universe.update()``` will be called that many times before rendering an image to the plot.

To investigate the accuracy of the simulation, there are a few methods that compute metrics of the universe. For example ```universe.calculate_momentum()```, ```universe.calculate_kinetic_energy()``` and ```universe.calculate_potential()```

### QuadTree Class
The ```BarnesHut``` algorithm has a ```QuadTree``` class attached, this contains the funtionality needed to insert, divide and traverse the quadtree structure used in the algorithm.

### Index Class
The ```FMM``` class takes advantage of the functionality of the ```Index``` class, each index represents the index of a cell in a quadtree. If contains an ```i``` and a ```j``` index, and a ```level```, representing the ```i```th row and ```j``` column of a quadtree at depth ```level```.
```Index``` classes have methods that relate to other indices in the quadtree, for example there are methods to return the ```parent()```, ```children()```, ```neighbours()``` and ```well_separated()``` cell indices. This is used extensively in the ```FMM``` algorithm.
