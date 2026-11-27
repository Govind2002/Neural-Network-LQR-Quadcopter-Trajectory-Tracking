# Neural Network-Augmented LQR for Quadcopter Trajectory Tracking

This project was developed as part of the Learning Based Control (EEL865) course at IIT Delhi. The aim is to track a quadcopter through multiple waypoints using a piecewise LQR controller augmented with an online adaptive neural network.

The reference trajectory is generated using minimum-jerk interpolation. The nonlinear quadcopter dynamics are simulated using SciPy, with PyBullet used for 3D visualization.

## Code Overview

- `main_simulation.py` – Runs the simulation and generates plots.
- `quadcopter_dynamics.py` – Nonlinear quadcopter model.
- `piecewise_linearisation.py` – System linearisation and LQR gain computation.
- `controller.py` – Combined LQR and neural network controller.
- `neural_network.py` – Neural network and online weight adaptation.
- `trajectory.py` – Reference trajectory generation.

## Installation and Usage

Install the required Python libraries:

```bash
pip install numpy scipy matplotlib pybullet
```

From the directory containing the Python source files, run:

```bash
python main_simulation.py
```

## Simulation

The simulation consists of a 38-second mission with five trajectory segments and external disturbances. It generates plots of the 3D trajectory, tracking errors, state evolution and control inputs.

Refer to `EndTerm_ProjectReport.pdf` for the mathematical formulation, controller design, stability analysis and simulation results.

## Collaborators

- S A Yogeswari
- Muhammed Aslam A
- Adithya Krishna
- Govind U

Claude AI (Anthropic) was used to assist with coding during the development of this project.
