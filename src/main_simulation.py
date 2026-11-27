
"""
simulation.py  –  Main simulation entry point
Point-to-point quadcopter tracking: Piecewise Linearisation + NN adaptive control

Modified for Spyder/GUI execution.
"""

import sys, argparse, time, os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pybullet as p
import pybullet_data

# Import existing modules (ensure these files are in the same directory)
from quadcopter_dynamics import QuadcopterParams, quadcopter_dynamics, linearise_at
from piecewise_linearisation import build_operating_regions, select_region
from controller import NNPiecewiseLQRController
from trajectory import MinJerkTrajectory
from neural_network import MLNN

# ── Mission definition ─────────────────────────────────────────────────────────
WAYPOINTS = [
    [0.0, 0.0, 0.0],
    [0.0, 2.0, 1.5],
    [2.0, 0.0, 1.5],
    [4.0, 2.0, 4.0],
    [0.0, 2.0, 1.0],
    [0.0, 5.0, 1.5],
]
SEG_TIMES = [6.0, 8.0, 8.0, 8.0, 8.0]   
DT        = 0.01
T_TOTAL   = sum(SEG_TIMES)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── PyBullet Setup (Modified for Spyder) ──────────────────────────────────────

def setup_pybullet(params):
    import pybullet as p
    import pybullet_data

    # 1. Handle Spyder's persistent kernel
    if p.isConnected():
        p.disconnect()

    # 2. Connect and set search path
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) 
    p.setGravity(0, 0, -params.g)
    p.loadURDF("plane.urdf")

    # 3. Create a Custom Drone with Propellers
    # Dimensions
    arm_length = 0.15
    prop_radius = 0.06
    prop_height = 0.01

    # Visual Shapes
    # Body Frame (X-shape)
    box_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[arm_length, 0.02, 0.01], rgbaColor=[0.2, 0.2, 0.2, 1])
    box_h = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, arm_length, 0.01], rgbaColor=[0.2, 0.2, 0.2, 1])
    
    # Propeller Shape (Thin cylinders)
    prop_v = p.createVisualShape(p.GEOM_CYLINDER, radius=prop_radius, length=prop_height, rgbaColor=[0.1, 0.1, 0.1, 0.6])

    # Combine into a MultiBody
    # We use link positions to offset the propellers to the ends of the arms
    link_Masses = [0.01, 0.01, 0.01, 0.01, 0.05] # 4 props + 1 cross bar
    link_CollisionShapeIndices = [-1] * 5
    link_VisualShapeIndices = [prop_v, prop_v, prop_v, prop_v, box_h]
    
    # Coordinates for propellers at the ends of the X-frame
    link_Positions = [
        [arm_length, 0, 0.02],   # Prop 1
        [-arm_length, 0, 0.02],  # Prop 2
        [0, arm_length, 0.02],   # Prop 3
        [0, -arm_length, 0.02],  # Prop 4
        [0, 0, 0]                # Cross bar
    ]
    link_Orientations = [[0, 0, 0, 1]] * 5
    link_InertialFramePositions = [[0, 0, 0]] * 5
    link_InertialFrameOrientations = [[0, 0, 0, 1]] * 5
    indices = [0] * 5 # Fixed to base
    joint_types = [p.JOINT_FIXED] * 5
    axis = [[0, 0, 0]] * 5

    drone_id = p.createMultiBody(
        baseMass=0.5,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=box_v,
        basePosition=[0, 0, 0.1],
        linkMasses=link_Masses,
        linkCollisionShapeIndices=link_CollisionShapeIndices,
        linkVisualShapeIndices=link_VisualShapeIndices,
        linkPositions=link_Positions,
        linkOrientations=link_Orientations,
        linkInertialFramePositions=link_InertialFramePositions,
        linkInertialFrameOrientations=link_InertialFrameOrientations,
        linkParentIndices=indices,
        linkJointTypes=joint_types,
        linkJointAxis=axis
    )

    # 4. Camera and Waypoints
    p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=45, cameraPitch=-30, 
                                 cameraTargetPosition=[0, 0, 1.0])

    for i, wp in enumerate(WAYPOINTS[1:]):
        vs_wp = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 0.7])
        p.createMultiBody(0, -1, vs_wp, wp)
        p.addUserDebugText(f"WP{i+1}", [wp[0], wp[1], wp[2] + 0.1], textColorRGB=[1, 1, 0])

    _draw_reference_preview(p)
    return {'p': p, 'drone_id': drone_id, 'prev_pos': None}

def _draw_reference_preview(p_client):
    traj = MinJerkTrajectory(WAYPOINTS, SEG_TIMES)
    pts = [traj.evaluate(tv)[0][0:3].tolist() for tv in np.linspace(0, T_TOTAL, 200)]
    for i in range(len(pts) - 1):
        p_client.addUserDebugLine(pts[i], pts[i + 1], lineColorRGB=[0.3, 0.3, 1.0], lineWidth=1.0)

def sync_pybullet(env, x):
    pos = x[0:3].tolist()
    quat = p.getQuaternionFromEuler([x[6], x[7], x[8]])
    p.resetBasePositionAndOrientation(env['drone_id'], pos, quat)
    p.stepSimulation()
    
    if env['prev_pos'] is not None:
        p.addUserDebugLine(env['prev_pos'], pos, lineColorRGB=[1, 0, 0], lineWidth=1.2)
    env['prev_pos'] = pos

# ── Simulation Logic ──────────────────────────────────────────────────────────

def simulate(use_pybullet=True, add_disturbance=True):
    params  = QuadcopterParams()
    traj    = MinJerkTrajectory(WAYPOINTS, SEG_TIMES)
    regions = build_operating_regions(WAYPOINTS[1:], params)
    ctrl    = NNPiecewiseLQRController(regions, nn_hidden=20)

    bullet_env = None
    if use_pybullet:
        bullet_env = setup_pybullet(params)

    x = np.zeros(12)
    rng = np.random.default_rng(0)
    t_log, x_log, xd_log, region_log, err_log, fhat_log,erratt_log =[], [], [], [], [], [], []

    print("Starting Simulation...")
    t = 0.0
    RENDER_EVERY = 5
    step_count = 0
    u_log = [] # Initialize this before the while loop
    while t <= T_TOTAL + DT / 2:
        x_des, _ = traj.evaluate(t)

        dist = np.zeros(12)
        if add_disturbance and 10.0 < t < 22.0:
            dist[3] += 0.4 * np.sin(2 * np.pi * 0.3 * t)
            dist[4] += 0.3 * rng.standard_normal() * 0.1

        u, info = ctrl.compute(t, x, x_des)

        def ode(tau, state):
            return quadcopter_dynamics(tau, state, u, params) + dist

        sol = solve_ivp(ode, [t, t + DT], x, method='RK45')
        x = sol.y[:, -1]
        x[2] = max(x[2], 0.0)

        t_log.append(t); x_log.append(x.copy()); xd_log.append(x_des.copy())
        region_log.append(info['region']); err_log.append(info['error_pos'])
        erratt_log.append(info['error_att'])
        fhat_log.append(info['f_hat_norm'])

        if bullet_env and (step_count % RENDER_EVERY == 0):
            sync_pybullet(bullet_env, x)
            time.sleep(DT * RENDER_EVERY * 0.5) # Speed factor
       
        u, info = ctrl.compute(t, x, x_des)
        u_log.append(u.copy()) # Log the 4 control inputs (thrust/moments)

        t += DT
        step_count += 1

    print("Simulation Finished.")
    plot_results(np.array(t_log), np.array(x_log), np.array(xd_log), np.array(erratt_log),
                 np.array(err_log), np.array(fhat_log), region_log,np.array(u_log))

def plot_results(t, x, xd,errat, err, fhat, region_log, u):
    
    # Create an independent 3D trajectory figure
    fig_3d = plt.figure(figsize=(10, 8))
    ax3 = fig_3d.add_subplot(1, 1, 1, projection='3d')

    # Plot the reference trajectory (dashed blue)
    ax3.plot(xd[:, 0], xd[:, 1], xd[:, 2], 'b--', lw=1.5, label='Reference', alpha=0.7)

    # Plot the actual flight path (solid red)
    ax3.plot(x[:, 0], x[:, 1], x[:, 2], 'r-', lw=1.0, label='Actual')
    
    # Highlight the waypoints (green spheres)
    # Note: 'wps' must be a numpy array of your WAYPOINTS
    wps = np.array(WAYPOINTS)
    ax3.scatter(wps[:, 0], wps[:, 1], wps[:, 2], c='g', s=60, zorder=5, label='Waypoints')

    # Formatting
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.set_zlabel('Z [m]')
    ax3.set_title('3D Flight Trajectory Tracking')
    ax3.legend(fontsize=9, loc='upper left')

    # Equalize axes (helpful for quadcopters to see realistic scale)
    # This prevents the trajectory from looking stretched
    max_range = np.array([x[:,0].max()-x[:,0].min(), x[:,1].max()-x[:,1].min(), x[:,2].max()-x[:,2].min()]).max() / 2.0
    mid_x = (x[:,0].max()+x[:,0].min()) * 0.5
    mid_y = (x[:,1].max()+x[:,1].min()) * 0.5
    mid_z = (x[:,2].max()+x[:,2].min()) * 0.5
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig("3D_Trajectory.png")
    # --- Figure 1: Evolution of All States ---
    fig_states, axs = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    fig_states.suptitle("Evolution of All Quadcopter States", fontsize=16)
    
    state_labels = [
        'X (m)', 'Y (m)', 'Z (m)', 
        'Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)',
        'Roll (rad)', 'Pitch (rad)', 'Yaw (rad)',
        'p (rad/s)', 'q (rad/s)', 'r (rad/s)'
    ]
    
    for i in range(12):
        row = i // 3
        col = i % 3
        axs[row, col].plot(t, x[:, i], label='Actual')
        if i < 3: # Add reference for XYZ position
            axs[row, col].plot(t, xd[:, i], 'r--', alpha=0.6, label='Desired')
        axs[row, col].set_ylabel(state_labels[i])
        axs[row, col].grid(True, alpha=0.3)
        if i == 0: axs[row, col].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("state_evolution.png")
    
    # --- Figure 2: Control Input Variation ---
    plt.figure(figsize=(10, 6))
    u_labels = ['Total Thrust ($f$)', 'Moment $M_x$', 'Moment $M_y$', 'Moment $M_z$']
    for i in range(4):
        plt.plot(t, u[:, i], label=u_labels[i])
    
    plt.title("Control Input Variation Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Input Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("control_inputs.png")
    
    plt.figure(figsize=(10, 8))

    # First subplot: Position Error
    plt.subplot(2, 1, 1) # 2 rows, 1 column, 1st plot
    plt.plot(t, err, label='Pos Error', color='blue')
    plt.title("Position Error Norm")
    plt.ylabel("Distance (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Second subplot: Attitude Error
    plt.subplot(2, 1, 2) # 2 rows, 1 column, 2nd plot
    plt.plot(t, errat, label='Att Error', color='red')
    plt.title("Attitude Error Norm")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("Error.png")
    
    print("Detailed plots saved:3D_Trajectory.png, state_evolution.png, control_inputs.png and Error.png")
# ── Main Entry ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Force defaults for easy "Run" in Spyder
    simulate(use_pybullet=True, add_disturbance=True)
    
