# Neural Network-Augmented LQR for Quadcopter Trajectory Tracking

A Learning Based Control (EEL865) end-term project, Indian Institute of Technology Delhi, April 2026.

This project implements point-to-point quadcopter trajectory tracking using piecewise linearisation, Linear Quadratic Regulators (LQR), and online neural network adaptation. The nonlinear plant is integrated using SciPy; PyBullet renders the simulated pose rather than serving as the plant dynamics solver.

![Historical end-term 3D tracking result](results/end_term/3D_Trajectory.png)

## Methodology

- Twelve-state nonlinear quadcopter model with four control inputs (total thrust and three body moments).
- Finite-difference linearisation around waypoint hover equilibria and continuous-time Riccati-based LQR design.
- Nearest-position operating-region selection.
- Quintic minimum-jerk reference trajectories between waypoints.
- Single-hidden-layer neural network with 20 hidden neurons, online weight updates, sigma-modification and weight clipping.
- NN warm-up, scaled adaptive contribution, and actuator input saturation.

The implemented control law is

$$u = u_{eq} - K_i(x-x_d) - \alpha_{NN}\hat f(x).$$

Here, the selected operating region determines the LQR gain, and the neural network supplies a bounded adaptive control contribution. The report develops a Lyapunov-based analysis; full switching stability and hardware validation remain open issues.

## Repository layout

```text
src/
  main_simulation.py           # End-term simulation and four plots
  simulation_comparison.py     # Later 38 s variant; comparison helpers
  simulation_19s.py            # Earlier archived 19 s variant
  quadcopter_dynamics.py       # Nonlinear model and Jacobians
  piecewise_linearisation.py   # Regions and LQR synthesis
  neural_network.py            # Forward pass and online adaptation
  controller.py                # Combined control law
  trajectory.py                # Minimum-jerk reference generator
docs/                          # Project reports
results/end_term/              # Historical end-term figures
results/comparison_archive/    # Historical comparison figures
outputs/                       # Intended location for future generated results (ignored)
requirements.txt
.gitignore
README.md
```

## Installation

Python 3.11 is recommended as the version cited in the end-term report. Exact dependency versions were not supplied. From the repository root, on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux/macOS, use `python3 -m venv .venv` and `source .venv/bin/activate` instead.

## Running simulations

From the repository root:

```bash
# End-term simulation (GUI enabled by default)
python src/main_simulation.py

# Later 38-second variant, without GUI
python src/simulation_comparison.py

# With visualization
python src/simulation_comparison.py --pybullet

# NN run without external disturbance
python src/simulation_comparison.py --no-disturbance

# Archived 19-second experiment
python src/simulation_19s.py
```

The supplied scripts are preserved without algorithmic or output-path changes. The main script saves plots relative to the working directory; the later comparison script saves its tracking plot beside the script. The intended `outputs/` layout is a future cleanup, not an implemented feature. The comparison helpers are not called by the default entry points; a normal run does not automatically generate a baseline comparison.

## Simulation configurations

| Parameter | End-term | Later comparison variant | Archived 19 s variant |
|---|---|---|---|
| Mission duration | 38 s | 38 s | 19 s |
| Step | 0.01 s | 0.01 s | 0.01 s |
| Segments | 5 | 5 | 5 |
| Disturbance interval | 10–22 s | 10–22 s | 6–12 s |
| NN learning rates (W, V) | 0.1, 0.1 | 1.0, 1.0 | 1.0, 1.0 |

The end-term mission starts at the origin and visits (0, 2, 1.5), (2, 0, 1.5), (4, 2, 4), (0, 2, 1), and (0, 5, 1.5), in metres. Segment durations are 6, 8, 8, 8 and 8 seconds. The plant mass is 0.5 kg. The default controller uses NN gain 0.3 and a two-second warm-up.

## Historical results

These figures are supplied project artifacts, not newly reproduced results.

![Position and attitude errors](results/end_term/Error.png)

[State evolution](results/end_term/state_evolution.png) · [Control inputs](results/end_term/control_inputs.png)

The earlier Word report records the following 19-second experiment metrics:

| Position error | Pure LQR | NN + LQR |
|---|---:|---:|
| Mean | 0.1270 m | 0.1496 m |
| Maximum | 0.2842 m | 0.2873 m |
| Final | 0.2295 m | 0.0575 m |

The report describes a 75% reduction in final error, not in mean or maximum error. These values are historical and have not been reproduced here. They must not be attributed to the newer 38-second configuration. The archived figures' exact run provenance is not independently verified.

![Historical comparison figure](results/comparison_archive/comparison_results.png)

## Reports

- [End-term project report (PDF)](docs/EndTerm_ProjectReport.pdf): model, control design, stability analysis and numerical results.
- [Earlier technical report (Word)](docs/quadcopter_nn_report.docx): 19-second study and comparative results.

## Known issues and future work

- Save figures before displaying them; the supplied main script can generate blank images.
- Evaluate the controller once per integration interval and log the input actually applied to the plant.
- Align state, reference and control timestamps.
- Add a dedicated comparison entry point and ensure identical disturbance, solver and mission settings for both controllers.
- Move generated plots to `outputs/`; record configuration and metrics with each run.
- Review the adaptation derivation, disturbance interpretation, saturation and switching assumptions before asserting stability guarantees.
- Add tests for hover equilibrium, trajectory endpoints, controller dimensions, saturation and finite simulation results.
- Rotor dynamics, aerodynamic drag and hardware validation are outside the current implementation.

## Contributors

As credited in the end-term report: S A Yogeswari, Muhammed Aslam A, Adithya Krishna, and Govind U. Original module author credits are retained in the code.

## License and publication

No license was supplied. Confirm permission from all contributors and the institution as applicable before publishing or choosing an open-source license. The reports and code may contain student identifiers; review and redact them as appropriate before making the repository public.
