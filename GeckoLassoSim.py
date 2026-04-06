# (1) TODO declaudify this code and push
# (2) TODO Make the visuals look good
# (3) TODO export sims and make the website look good!!

"""
Gecko Lasso Simulation — simulation of a cable wrapping, capturing, and detumbling a satellite
M. Coughlin 2026 created for the BDML research group at Stanford University

This code is designed to use the MuJoCo native physics engine as much as possible

Requirements:
    pip install mujoco numpy matplotlib trimesh pyobjc-framework-Quartz imageio av

    pyobjc-framework-Quartz, imageio, and av are only required to record videos of the simulation

Run:
    python GeckoLassoSim.py
"""
from datetime import datetime
from math import sqrt
from pathlib import Path
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import trimesh


#########
# CONFIGURATION
# These variables control visuals and computational optimizations, not the physical parameters in the simulation
#########

SHOW_INACTIVE_SEG = False  # Whether to show inactive nodes in the viewer (a line of nodes behind the AV)
CABLE_VISUAL_RADIUS = 0.07  # Size of the cable connecting the point masses on the display. Not physical. Units of m.
CABLE_NODE_VISUAL_RADIUS = 0.05  # Radius of the point masses on the cable on the display. Not physical! Units of m.

# Node and cable colors (RGBA)
COLOR_FROZEN = [1, 0, 0, 1]           # Red — frozen/welded nodes
COLOR_SPOOL = [0, 1, 0, 1]            # Green — spool node
COLOR_DAMPED = [0, 0.8, 1, 1]         # Cyan — damped nodes
COLOR_UNDAMPED = [1, 1, 0, 1]         # Yellow — undamped nodes
COLOR_FREE_LINK = [1.0, 0.4, 0.15, 1.0]    # Orange — free link from AV to spool
COLOR_FROZEN_SEG = [1.0, 0.4, 0.15, 1.0]  # [0.8, 0.1, 0.1, 1.0]    # Dark red — frozen chain segments
COLOR_ACTIVE_SEG = [1.0, 0.45, 0.15, 1.0]  # Orange — active chain segments
COLOR_INACTIVE = [0.5, 0.5, 0.5, 0]        # Invisible gray — inactive/pre-spawned nodes
COLOR_SATELLITE = [0.78, 0.82, 0.88, 1]    # Light gray-blue — satellite material
COLOR_AV = [0.3, 0.55, 0.95, 1]            # Blue — AV material
COLOR_GROUND = [0.2, 0.2, 0.2, 1]          # Dark gray — ground plane material

# Density of the satellite material. Right now, the density is homogenous. Non-homogenous densities are possible but
# require manually calculating the inertial tensor and center of mass and specifying them in the XML
SAT_MATERIAL_DENSITY = 1000  # Density of the satellite (kg/m^3). 7000 corresponds to aluminum.

# Contact solver parameters for cable-satellite collisions
CONTACT_MARGIN = 0.001        # Distance (m) at which contact constraints activate before geometric penetration
CONTACT_SOLREF_DAMPRATIO = 1  # Damping ratio for contact spring for contact forces (1 = critically damped)
CONTACT_SOLREF_STIFFNESS = 0.4  # Dimensionless constant relating contact timeconst to sqrt(mass/tension)

# (dmin, dmax, width, midpoint, power) — constant impedance
# dmin == dmax so the behavior scales properly regardless of cable tension (ie with proper values, the simulation will
# remain stable and well-behaves over tensions spanning at least 2 orders of magnitude
CONTACT_SOLIMP = (0.95, 0.95, 0.001, 0.5, 2)

# Capstan friction correction: MuJoCo's friction underestimates the continuous capstan friction effect
# (T_high / T_low = e^(mu * theta)). When enabled, an external corrective friction force is applied to
# each cable node in contact with the satellite to make up the deficit.
# IMPORTANT NOTES: The capstan correction has several limitations:
#      - It is only designed to work with cylindrical and oval-like geometries
#      - The geometric axis of the cylinder must be aligned with its z-axis
#      - It assumes a massless cable

CAPSTAN_CORRECTION = False

# Whether to recompile the simulation to add and freeze nodes
#   False:  spawn all nodes at the start of the simulation and solve the full simulation at every time step. Simplest
#           logic, most computationally intensive
#   True:   Recognizing that only a few cable nodes are actually doing interesting things at each time step, only solve
#           the simulation for these nodes. The remaining nodes receive the following treatment: inactive nodes are
#           left stationary and motionless. Nodes on the satellite which are not moving relative to the satellite are
#           welded to the satellite and treated as moving exactly with that body. Note that the mujoco simulation solved
#           for the state of the "interesting" nodes is NOT the simulation which is displayed to the user. The state of
#           the "interesting" nodes and satellite is imported into another mujoco model, used for display only, that
#           includes all bodies. The state of the remaining "uninteresting" nodes can be found very quickly using the
#           model described above — no need to run a full physics simulation!
ENABLE_RECOMPILE = True

# The number of inactive segments to add to the model upon recompilation. Lower values lead to more frequent
# recompilation, but keep the models smaller. Larger values reduce the number of recompilations, but each time step
# takes longer to solve because of the added degrees of freedom.
RECOMPILE_BUFFER = 3

# The number of nodes, as measured from the last node touching the satellite, to leave undamped in the lateral
# direction. Note that axial damping is always applied to all tendons.
UNDAMPED_LATERAL_NODES_BEFORE_SAT = 4

# The desired ratio between the length of the free link and the total distance between the AV and the current departure
# point on the satellite
FREE_LINK_AV_SAT_DIST_RATIO = 0.6

# A new node spawns when the free link length is equal to the target free link length plus the equilibrium spacing
# between nodes multiplied by SPAWN_NEW_NODE_GROWTH_THRESHOLD. This allows for decoupling node spawn and despawn
# position, preventing nodes from repeatedly spawning and despawning due to oscillations around the spawn/despawn
# position. This should be >= 1 to be meaningful — otherwise, the spool will move further back than desired on the
# free link!
SPAWN_NEW_NODE_GROWTH_THRESHOLD = 1.1

# A similar parameter to SPAWN_NEW_NODE_GROWTH_THRESHOLD, except this time controlling when a node despawns. To be
# meaningful, it must be <= 1
DESPAWN_NODE_SHRINK_THRESHOLD = 0.1

# Grace period (seconds) for the departure node: if a node was the departure point and loses contact,
# it is still treated as the departure point for this duration before falling back to the next contact.
DEPARTURE_GRACE_PERIOD = 0.2

# Recording settings — captures OS-level screenshots at the native screen resolution and create a video
# This is the best way to get high quality exports on MacOS
RECORD = True
RECORD_FPS = 30               # Frame rate in simulation time (frames per sim-second)
RECORD_OUTPUT = "Simulation " + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"


def _rgba_str(color):
    """Convert an RGBA color list to a MuJoCo XML rgba string."""
    return f"{color[0]} {color[1]} {color[2]} {color[3]}"


def _load_mesh(path: Path) -> Path:
    """Load a .obj mesh, make it watertight, center it on its center of mass, and export a cleaned copy for MuJoCo.
    This is necessary since .obj files exported from CAD software are not guaranteed to be "well-behaved".

    Args:
        path: Filesystem path to the source .obj file.

    Returns:
        Path to the cleaned, re-exported .obj file, written as a hidden file (on Mac).
    """
    mesh = trimesh.load(path, force="mesh")

    # Ensure mesh is watertight, which is required for MuJoCo to calculate mass properties correctly.
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())

    # Place simulation origin at center of mass of satellite
    mesh.apply_translation(-mesh.center_mass)

    temp_export_file = path.parent / ("._Repositioned_" + path.name)
    mesh.export(temp_export_file)
    return temp_export_file


class Simulation:
    def __init__(self, sat_omega: float = 1, sat_rotation_axis: tuple = (0, 0.02, 1),
                 sat_attach_pos: tuple = (-1.8, 0, 0), av_init_pos: tuple = (-4.8, 15, 0),
                 thruster_react_func = lambda torque_vec: -1 * torque_vec, cable_tension: float = 500,
                 cable_stiffness: float = 7000, axial_damping_ratio: float = 1, lateral_damping_ratio: float = 0.7,
                 cable_pt_mass: float = 0.000259, cable_node_radius: float = 0.005,
                 cable_friction: tuple = (0.8, 0.2, 0.2), capstan_friction = 0.5, cable_seg_len: float = 0.5,
                 free_link_cable_ratio: float = 0.7, max_seg_num: int = 1800, time_step: float = 0.00005,
                 imp_ratio: float = 10, freeze_speed: float = 0.0075, freeze_time: float = 0.1,
                 freeze_sample_interval: float = 0.002, freeze_max_tension_frac: float = float('inf'),
                 capstan_recompute_dt: float = 0.002, end_simulation_time: float = 60, record_output: str = None,
                 sat_obj_path: str = "Assets/ShortRocketBody.obj"):
        """Initialize the simulation, compile the MuJoCo physics and display models, and set initial conditions for the
        satellite and cable.

        Args:
            sat_omega: Initial satellite angular velocity (rad/s).
            sat_rotation_axis: Axis about which the satellite initially spins, in the coordinate frame of the satellite
                mesh.
            sat_attach_pos: Position where the cable attaches to the satellite, as a displacement vector from the
                satellite's center of mass.
            av_init_pos: Initial position of the Autonomous Vehicle (AV), as a displacement vector from the satellite's
                center of mass.
            thruster_react_func: Callable that takes the cable tension vector and returns the AV thruster reaction force
                vector.
            cable_tension: Constant tension applied along the free link (N).
            cable_stiffness: Spring stiffness of each inter-node tendon (N/m).
            axial_damping_ratio: Ratio of applied axial damping to critical damping. Values > 1 are over-damped; < 1 are
                under-damped.
            lateral_damping_ratio: Ratio of applied lateral damping (orthogonal to cable axis) to critical damping.
            cable_pt_mass: Mass of each discretized cable node (kg). Default values correspond to a nylon cable with
                density ~1100 kg/m^3 and ~1 mm diameter.
            cable_node_radius: Radius of each discretized cable node for physical simulation (m).
            cable_friction: (sliding, torsional, rolling) friction coefficients between cable nodes and the satellite
                surface.
            capstan_friction: The friction coefficient used when forcing the tension in the cable to exponentially decay
                due to the capstan effect. Only used when CAPSTAN_CORRECTION = True. Physically, this value is expected
                to bebe cable_friction[0], but cable_friction may not hold physical value to improve cable adhesion to
                the satellite.
            cable_seg_len: Unstretched rest length between adjacent cable nodes (m).
            free_link_cable_ratio: Minimum free-link length as a fraction of the anchor-to-AV distance. May be
                overridden by min_nodes_before_sat.
            max_seg_num: Maximum number of cable nodes allowed in the simulation.
            time_step: Physics integration timestep (s). Values in the ~10-50 us range work well.
            imp_ratio: MuJoCo impedance ratio for friction constraint solving.
            freeze_speed: A node's relative speed must stay below this threshold (m/s) for the entire freeze_time window
                to be frozen to the satellite.
            freeze_time: Duration (s) a node must remain below freeze_speed to be frozen.
            freeze_sample_interval: Time between node speed measurements used for freeze detection (s).
            freeze_max_tension_frac: Maximum tension as a fraction of cable_tension for a node to be eligible for
                freezing. This can ensure a node which may temporarily pause can move again if its motion is influenced
                by capstan friction.
            capstan_recompute_dt: Maximum time (s) between recomputes of the capstan friction correction forces.
                The correction is applied every timestep but only recomputed at this interval. Smaller values are more
                accurate; larger values are faster. Set to 0 to recompute every step.
            end_simulation_time: Maximum simulation duration in simulation time (s).
            record_output: Output file path for recorded video. Defaults to a timestamped filename in the current
                directory. Only used when RECORD = True.
            sat_obj_path: Path to the satellite .obj mesh file, relative to the current working directory.
        """

        # Save parameters for the simulation
        self.sat_omega = sat_omega
        self.sat_rotation_axis = np.array(sat_rotation_axis)
        self.sat_attach_pos = np.array(sat_attach_pos)
        self.av_init_pos = np.array(av_init_pos)
        self.thruster_react_func = thruster_react_func
        self.cable_tension = cable_tension
        self.cable_stiffness = cable_stiffness

        # Critical damping is different for small perturbations in the axial direction and in the transverse direction.
        # Further, since transverse damping is applied directly at nodes, a factor of 2 is required. MuJoCo applies the
        # axial damping twice at each node, once for each tendon, no factor of 2 is required.
        critical_damping_axial = np.sqrt(2.0 * cable_stiffness * cable_pt_mass)
        critical_damping_transverse = 2.0 * np.sqrt(2.0 * cable_pt_mass * cable_tension * cable_stiffness /
                                                    (cable_stiffness * cable_seg_len + cable_tension))
        self.cable_axial_damping = axial_damping_ratio * critical_damping_axial
        self.cable_lateral_damping = lateral_damping_ratio * critical_damping_transverse

        self.cable_mass = cable_pt_mass
        self.cable_node_radius = cable_node_radius
        self.cable_friction = cable_friction
        self.capstan_friction = capstan_friction
        self.capstan_recompute_dt = capstan_recompute_dt
        self.cable_seg_len = cable_seg_len
        self.max_seg_num = max_seg_num
        self.time_step = time_step
        self.imp_ratio = imp_ratio

        self._freeze_threshold = freeze_speed
        # The number of measurements which must be below freeze_speed to freeze the node
        self._freeze_window_samples = int(freeze_time / freeze_sample_interval) + 1
        self._freeze_check_interval = freeze_sample_interval
        self._freeze_max_tension_frac = freeze_max_tension_frac

        self.end_simulation_time = end_simulation_time

        # Calculate the equilibrium length of the string
        equilibrium_cable_stretch = self.cable_tension / self.cable_stiffness
        self.seg_equilibrium_len = self.cable_seg_len + equilibrium_cable_stretch

        # Contact spring timeconst scales as sqrt(mass/tension) to maintain consistent penetration depth across
        # different tension and mass configurations
        contact_timeconst = CONTACT_SOLREF_STIFFNESS * np.sqrt(cable_pt_mass / cable_tension)
        self.contact_solref = (contact_timeconst, CONTACT_SOLREF_DAMPRATIO)

        self.record_output = record_output if record_output is not None else (
            "Simulation " + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4")

        # Cache mesh files to reduce computational cost of a model recompilation
        cwd = Path.cwd()
        self._sat_obj_file = _load_mesh(cwd / sat_obj_path)
        self._av_obj_file = _load_mesh(cwd / "Assets" / "AV_1_0p5_0p5.obj")
        self._background_im_file = str(cwd / "Assets" / "BackgroundNASA.png")

        # Compute initial segment count and direction (needed before first compile)
        anchor_av_init_dir = self.av_init_pos - self.sat_attach_pos
        init_cable_max_len = np.linalg.norm(anchor_av_init_dir)
        anchor_av_init_dir /= init_cable_max_len

        # Chain covers (1 - free_link_fraction) of the initial cable length.
        # The free link is the first connecting the AV to the rest of the chain
        self.num_init_segments = int(init_cable_max_len * (1 - free_link_cable_ratio) / self.seg_equilibrium_len) + 1

        if ENABLE_RECOMPILE:
            # Compile only enough segments for the initial state plus buffer
            self.compiled_seg_count = min(self.num_init_segments + RECOMPILE_BUFFER, self.max_seg_num)
        else:
            # Load all nodes
            self.compiled_seg_count = self.max_seg_num

        # Current anchor position in satellite body frame. persists across recompiles
        self._current_anchor_body_pos = np.array(self.sat_attach_pos, dtype=np.float64)

        # Creates the physics model
        self.xml = self._create_model_xml(self.compiled_seg_count)
        self.model = mujoco.MjModel.from_xml_string(self.xml, None)
        self.data = mujoco.MjData(self.model)

        # Create lightweight display model (satellite + AV only, no cable bodies or tendons)
        display_xml = self._create_model_xml(0)
        self.display_model = mujoco.MjModel.from_xml_string(display_xml, None)
        self.display_data = mujoco.MjData(self.display_model)
        # Frame axes scale with stat.extent — override so they aren't enormous
        self.display_model.stat.extent = 5.0

        # Build ID caches for physics model
        self._rebuild_id_caches()

        # Display model caches (satellite only — cable nodes are rendered as scene geoms)
        self._display_cylinder_jnt_id = mujoco.mj_name2id(self.display_model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_rotation")


        # Save the ID for the satellite
        self._phys_cyl_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
        self._disp_cyl_body = mujoco.mj_name2id(self.display_model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")

        # Give the satellite the proper initial conditions
        qvel_addr = self.model.jnt_dofadr[self.cylinder_jnt_id]
        sat_rot_dir = self.sat_rotation_axis / np.linalg.norm(self.sat_rotation_axis)
        self.data.qvel[qvel_addr + 3:qvel_addr + 6] = self.sat_omega * sat_rot_dir

        # Give the cable an initial velocity
        # This makes the model work slightly better since it reduces the sudden tension spike at the start
        # anchor_av_init_dir is fine to use as the direction here since there is a dot product — ie it is not off by -1
        init_vel_anchor = np.cross(self.sat_omega * sat_rot_dir, self.sat_attach_pos)

        # Project anchor velocity onto cable direction - this is the initial speed of all nodes except the anchor
        cable_axial_velocity = (init_vel_anchor @ anchor_av_init_dir) * anchor_av_init_dir

        for string_node in range(self.num_init_segments):
            body_id = self.cable_jnt_ids[string_node]
            qvel_addr = self.model.jnt_dofadr[body_id]
            self.data.qvel[qvel_addr:qvel_addr + 3] = cable_axial_velocity

        # The anchor additionally has the orthogonal component since it is on the rotating body
        qvel_addr = self.model.jnt_dofadr[self.cable_jnt_ids[0]]
        self.data.qvel[qvel_addr:qvel_addr + 3] = init_vel_anchor

        # Forward pass to initialize physics state consistently
        mujoco.mj_forward(self.model, self.data)

        # Compute body masses for display on the sidebar
        sat_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
        self.sat_mass = self.model.body_subtreemass[sat_body_id]
        self.av_mass = self.model.body_subtreemass[self.av_body_id]

        # Track the current thruster reaction force for overlay display
        self.current_thruster_force = np.zeros(3)

        # Deactivate tendons for inactive segments by setting stiffness to 0
        for i in range(self.num_init_segments, self.compiled_seg_count):
            tendon_id = self.cable_tendon_ids[i]
            self.model.tendon_stiffness[tendon_id] = 0
            self.model.tendon_damping[tendon_id] = 0

        # Store the state of the cable unspooling
        self.spool_idx = self.num_init_segments - 1
        self.active_count = self.num_init_segments

        self.anchor_idx = 0  # Absolute index of the first node in the physics model
        self._frozen_body_frame_pos = []  # List of satellite-body-frame positions of frozen nodes on the satellite
        self._frozen_inter_angles = []   # Per-frozen-node inter-node angle (radians) for capstan force computation
        self._frozen_wrap_total = 0.0    # Cached sum of _frozen_inter_angles, updated on freeze events
        self._freeze_vel_history = {}  # Dict of physics idx: deque of recent relative velocities

        self._next_freeze_check = 0
        self._pending_freeze_count = 0
        # Physics index of the node where damping begins. This is after the handful of nodes coming off the satellite
        # which are undamped in the plane not parallel to the cable axis
        self._damping_start = 0

        # Tension tracking
        # This is mainly for debugging, since issues tend to show up clearly in the cable tension
        self.tension_history = []
        self.target_history = []
        self.time_history = []
        self._tension_record_interval = 1.0 / 200.0
        self._next_tension_record_time = 0.0

        # Contact force diagnostics for debugging capstan friction behavior
        self.contact_diag_history = []
        self._sat_force_snapshot = {
            'constraint': np.zeros(3), 'passive': np.zeros(3), 'applied': np.zeros(3)
        }
        self._cable_sat_contacts_cache = None

        self.sync_display()

    def _rebuild_id_caches(self):
        """Populate MuJoCo ID lookup tables and pre-computed arrays after a model (re)compilation.
        Also initializes per-step caches used by _get_cable_leave_pos."""
        # Given the XML structure many indices can be found by using the fact cable node indices are sequential
        # However, this function runtime is insignificant in terms of the simulation, but the code is left as-is for
        # readability and durability against changes in XML structure
        self.cylinder_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_rotation")
        self.av_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "av_body")
        self.attachment_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_point")

        self._phys_cable_body_arr = np.empty(self.compiled_seg_count, dtype=np.intp)
        self._phys_cable_geom_arr = np.empty(self.compiled_seg_count, dtype=np.intp)
        cable_tendon_ids_list = []
        self.cable_jnt_ids = []
        for i in range(self.compiled_seg_count):
            self._phys_cable_body_arr[i] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"cable_{i}")
            self._phys_cable_geom_arr[i] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"cable_geom_{i}")
            self.cable_jnt_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cable_free_{i}"))

            cable_tendon_ids_list.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, f"cable_tendon_{i}"))
        self.cable_tendon_ids = np.array(cable_tendon_ids_list, dtype=np.intp)

        self._phys_cyl_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")

        # Pre-computed DOF addresses for vectorized force application
        self._cable_dof_adrs = np.array([self.model.jnt_dofadr[jid] for jid in self.cable_jnt_ids], dtype=np.intp)
        self._sat_geom_id = self.model.body_geomadr[self._phys_cyl_body]

        # Reverse lookup: geom ID -> cable index for _find_cable_sat_contacts
        self._geom_id_to_cable_idx = {gid: i for i, gid in enumerate(self._phys_cable_geom_arr)}

        # Pre-allocated arrays for _apply_tension_and_forces, reducing per-step allocations
        self._torque_vec = np.zeros(3, dtype=np.float64)
        self._qfrc_buf = np.zeros(self.model.nv, dtype=np.float64)
        self._contact_force_buf = np.zeros(6, dtype=np.float64)

        # Per-step caches
        self._cable_leave_idx = 0
        self._cable_leave_grace_expiry = 0.0  # sim time when the grace period expires
        self._cable_sat_contacts_cache = None

        # Capstan correction cache: stores the qfrc_applied delta from the capstan block so it can be
        # replayed on steps where the correction is not recomputed.
        self._capstan_qfrc_delta = np.zeros(self.model.nv, dtype=np.float64)
        self._next_capstan_recompute = 0.0

    def _recompile_model(self, new_seg_count: int, anchor_body_frame_pos=None, frozen_this_recompile: int = 0):
        """Rebuild the MuJoCo model with a new number of cable segments, preserving satellite and active cable state
            (positions, velocities, tendon properties) across the recompilation.

        Args:
            new_seg_count: Number of cable segments in the recompiled model.
            anchor_body_frame_pos: Satellite-body-frame position for the cable anchor site. If None, the current anchor
                position is reused.
            frozen_this_recompile: Number of nodes at the anchor end that were just frozen and should be skipped when
                copying state from the old model.
        """
        old_model = self.model
        old_data = self.data
        old_compiled = self.compiled_seg_count
        old_sat_jnt_id = self.cylinder_jnt_id
        old_cable_jnt_ids = self.cable_jnt_ids

        # Build the new model while old_model/old_data are still alive
        self.compiled_seg_count = new_seg_count
        self.xml = self._create_model_xml(new_seg_count, anchor_body_frame_pos=anchor_body_frame_pos)
        self.model = mujoco.MjModel.from_xml_string(self.xml, None)
        self.data = mujoco.MjData(self.model)
        self._rebuild_id_caches()

        # Copy simulation time
        self.data.time = old_data.time

        # Copy satellite state into the new model
        old_sat_qpos_adr = old_model.jnt_qposadr[old_sat_jnt_id]
        old_sat_qvel_adr = old_model.jnt_dofadr[old_sat_jnt_id]
        new_sat_qpos_adr = self.model.jnt_qposadr[self.cylinder_jnt_id]
        new_sat_qvel_adr = self.model.jnt_dofadr[self.cylinder_jnt_id]
        self.data.qpos[new_sat_qpos_adr:new_sat_qpos_adr + 7] = old_data.qpos[old_sat_qpos_adr:old_sat_qpos_adr + 7]
        self.data.qvel[new_sat_qvel_adr:new_sat_qvel_adr + 6] = old_data.qvel[old_sat_qvel_adr:old_sat_qvel_adr + 6]

        # Copy active cable segment states into the recompiled model, skipping frozen nodes.
        # Old physics index (i + frozen_this_recompile) maps to new physics index i.
        num_to_copy = min(self.active_count, old_compiled - frozen_this_recompile)
        for i in range(num_to_copy):
            old_id = old_cable_jnt_ids[i + frozen_this_recompile]
            old_qpos_adr = old_model.jnt_qposadr[old_id]
            old_qvel_adr = old_model.jnt_dofadr[old_id]

            new_qpos_adr = self.model.jnt_qposadr[self.cable_jnt_ids[i]]
            new_qvel_adr = self.model.jnt_dofadr[self.cable_jnt_ids[i]]

            self.data.qpos[new_qpos_adr:new_qpos_adr + 7] = old_data.qpos[old_qpos_adr:old_qpos_adr + 7]
            self.data.qvel[new_qvel_adr:new_qvel_adr + 6] = old_data.qvel[old_qvel_adr:old_qvel_adr + 6]

        # Deactivate tendons for segments beyond active count
        for i in range(self.active_count, self.compiled_seg_count):
            tid = self.cable_tendon_ids[i]
            self.model.tendon_stiffness[tid] = 0
            self.model.tendon_damping[tid] = 0

        # Forward the changes through the model
        mujoco.mj_forward(self.model, self.data)

        # Print status update
        print(f"  [RECOMPILE] {old_compiled} -> {new_seg_count} segments (active: {self.active_count})")

    def sync_display(self):
        """Update the satellite pose in the display model and compute cable node render data.

        Returns:
            A tuple of (positions, colors) where positions is an (N, 3) float64 array of world-frame node positions and
            colors is an (N, 4) float32 RGBA array, both ordered by absolute cable index, meaning frozen nodes come
            first, then active nodes follow
        """
        # Update satellite position in the display model
        phys_cyl_qadr = self.model.jnt_qposadr[self.cylinder_jnt_id]
        disp_cyl_qadr = self.display_model.jnt_qposadr[self._display_cylinder_jnt_id]
        self.display_data.qpos[disp_cyl_qadr:disp_cyl_qadr + 7] = self.data.qpos[phys_cyl_qadr:phys_cyl_qadr + 7]
        mujoco.mj_kinematics(self.display_model, self.display_data)

        # +1 for the attachment point, which is the first entry
        total_visible = 1 + self.anchor_idx + self.active_count
        positions = np.empty((total_visible, 3), dtype=np.float64)
        colors = np.empty((total_visible, 4), dtype=np.float32)

        # Satellite pos for body-frame → world-frame transforms
        sat_pos = self.data.xpos[self._phys_cyl_body]
        sat_quat = self.data.xquat[self._phys_cyl_body]
        world_pos = np.zeros(3)

        mujoco.mju_rotVecQuat(world_pos, self.sat_attach_pos, sat_quat)
        positions[0] = world_pos + sat_pos
        colors[0] = COLOR_FROZEN

        # Frozen nodes: transform stored body-frame positions to world frame
        for f_idx in range(self.anchor_idx):
            mujoco.mju_rotVecQuat(world_pos, self._frozen_body_frame_pos[f_idx], sat_quat)
            positions[1 + f_idx] = world_pos + sat_pos
            colors[1 + f_idx] = COLOR_FROZEN

        # Active nodes: read positions from physics qpos
        for i in range(self.active_count):
            p_adr = self.model.jnt_qposadr[self.cable_jnt_ids[i]]
            abs_idx = 1 + self.anchor_idx + i
            positions[abs_idx] = self.data.qpos[p_adr:p_adr + 3]

            if i == self.spool_idx:
                colors[abs_idx] = COLOR_SPOOL
            elif i >= self._damping_start:
                colors[abs_idx] = COLOR_DAMPED
            else:
                colors[abs_idx] = COLOR_UNDAMPED

        return positions, colors

    def _create_model_xml(self, num_segments, anchor_body_frame_pos=None):
        """Generate the MuJoCo XML string for the full simulation scene including the satellite, AV, and optionally
        cable bodies with free joints and spatial tendons, which are only used in the physics model. Note that this
        creates an XML for a system in equilibrium — if the state needs to be imported from a previous model, that
        must be done after loading the XML into MuJoCo!

        Pass num_segments=0 to create a display-only model (satellite and AV, no cable bodies or tendons).

        Args:
            num_segments: Total number of cable body segments to include in the XML.
            anchor_body_frame_pos: Satellite-body-frame position of the cable anchor site. If None, the current anchor
                position (_current_anchor_body_pos) is used. This is either the initial anchor point or the last frozen
                node.
        """
        # Anchor position in satellite body frame (where cable_0 attaches)
        attach_pos = anchor_body_frame_pos if anchor_body_frame_pos is not None else self._current_anchor_body_pos

        # Vector from attachment toward AV
        anchor_av_dir = self.av_init_pos - self.sat_attach_pos
        init_cable_max_len = np.linalg.norm(anchor_av_dir)
        anchor_av_dir /= init_cable_max_len

        # Calculate positions for all nodes at equilibrium spacing along the cable direction.
        offsets = (np.arange(num_segments) + 1) * self.seg_equilibrium_len
        node_locations = self.sat_attach_pos + np.outer(offsets, anchor_av_dir)

        # Build cable bodies with sites
        cable_bodies = ""
        cable_tendons = ""
        for i in range(num_segments):
            pos = node_locations[i]
            color = _rgba_str(COLOR_UNDAMPED) if i < self.num_init_segments else _rgba_str(COLOR_INACTIVE)

            cable_bodies += f"""
            <body name="cable_{i}" pos="{pos[0]} {pos[1]} {pos[2]}">
                <freejoint name="cable_free_{i}"/>
                <geom name="cable_geom_{i}" type="sphere" size="{self.cable_node_radius}" mass="{self.cable_mass}"
                      friction="{self.cable_friction[0]} {self.cable_friction[1]} {self.cable_friction[2]}"
                      rgba="{color}" condim="6" contype="1" conaffinity="2"/>
                <site name="cable_site_{i}" pos="0 0 0" size="{self.cable_node_radius}"/>
            </body>
            """

        # Build tendons: anchor tendon (attachment_point → cable_site_0) plus inter-node tendons.
        # All tendons share the same springlength, stiffness, and damping
        if num_segments > 0:
            # Anchor tendon connects the satellite attachment site to the first cable node
            from_site = "attachment_point"
            for i in range(num_segments):
                to_site = f"cable_site_{i}"
                cable_tendons += f"""
                <spatial name="cable_tendon_{i}" springlength="{self.cable_seg_len}"
                        stiffness="{self.cable_stiffness}" damping="{self.cable_axial_damping}" limited="false">
                    <site site="{from_site}"/>
                    <site site="{to_site}"/>
                </spatial>"""
                from_site = to_site

        # Next build the main XML defining the simulation
        xml = f"""
        <mujoco model="Gecko Lasso Simulation">
            <option gravity="0 0 0" timestep="{self.time_step}" integrator="implicit" 
                cone="elliptic" impratio="{self.imp_ratio}"/>

            <visual>
                <quality shadowsize="4096"/>
                <headlight diffuse="0.5 0.5 0.5" specular="0.2 0.2 0.2" ambient="0.2 0.2 0.2"/>
            </visual>

            <asset>
                <texture type="skybox" file="{self._background_im_file}" rgb1="0.1 0.1 0.2" rgb2="0.3 0.3 0.4"/>

                <material name="sat_mat" rgba="{_rgba_str(COLOR_SATELLITE)}" emission="0.35" specular="0.3" shininess="0.15"/>
                <material name="av_mat" rgba="{_rgba_str(COLOR_AV)}" emission="0.3" specular="0.3" shininess="0.15"/>
                <material name="ground_mat" rgba="{_rgba_str(COLOR_GROUND)}" reflectance="0.1"/>
                <mesh name="sat_mesh" file="{self._sat_obj_file}" inertia="exact"/>
                <mesh name="av_mesh" file="{self._av_obj_file}" inertia="exact"/>
            </asset>

            <worldbody>
                <!--geom type="plane" size="20 20 0.1" pos="0 0 -10" material="ground_mat" contype="0" conaffinity="0"/-->

                 <body name="av_body" pos="{self.av_init_pos[0]} {self.av_init_pos[1]} {self.av_init_pos[2]}">
                    <!--freejoint name="av_free_joint"/-->
                    <geom type="mesh" mesh="av_mesh" contype="4" conaffinity="2" friction="0 0 0" condim="6" material="av_mat"/>
                    <site name="cable_origin" pos="0 0 0" size="{self.cable_node_radius}" rgba="{_rgba_str(COLOR_SPOOL)}"/>
                </body>

                <body name="cylinder" pos="0 0 0">
                    <freejoint name="cylinder_rotation"/>
                    <geom type="mesh" mesh="sat_mesh" contype="2" conaffinity="1"
                        density="{SAT_MATERIAL_DENSITY}" friction="{self.cable_friction[0]} {self.cable_friction[1]} 
                        {self.cable_friction[2]}" condim="6" material="sat_mat" margin="{CONTACT_MARGIN}"
                        solimp="{CONTACT_SOLIMP[0]} {CONTACT_SOLIMP[1]} {CONTACT_SOLIMP[2]} {CONTACT_SOLIMP[3]}
                                {CONTACT_SOLIMP[4]}"
                        solref="{self.contact_solref[0]} {self.contact_solref[1]}" priority="1"/>

                    <site name="attachment_point" pos="{attach_pos[0]} {attach_pos[1]} {attach_pos[2]}"
                        size="{self.cable_node_radius}" rgba="{_rgba_str(COLOR_FROZEN)}"/>
                </body>

            {cable_bodies}

            </worldbody>
            {"" if num_segments == 0 else f'''
            <tendon>
                {cable_tendons}
            </tendon>
            '''}
        </mujoco>
        """

        return xml

    def _get_node_pos(self, idx) -> np.ndarray:
        """Return the world-frame position of the cable body at the given physics index.

        Args:
            idx: Physics-model cable index (0 = first active node at the anchor end).
        """
        return self.data.xpos[self._phys_cable_body_arr[idx]]

    def _find_cable_sat_contacts(self):
        """Identify which cable nodes are in contact with the satellite.

        Returns:
            cable_node_indices: Sorted list of cable node indices in contact with the satellite.
            contact_ci_by_node: Dict mapping cable_idx to list of MuJoCo contact indices (ci) for that node.
        """
        
        if self._cable_sat_contacts_cache is not None:
            return self._cable_sat_contacts_cache

        if self.data.ncon == 0:
            self._cable_sat_contacts_cache = ([], {})
            return self._cable_sat_contacts_cache

        # Vectorized: read all contact geom pairs at once. This is a pre-allocated array by MuJoCo, so it must be sliced
        # here for use
        geom1 = self.data.contact.geom1[:self.data.ncon]
        geom2 = self.data.contact.geom2[:self.data.ncon]

        # Find contacts where one geom is the satellite
        sat_in_g1 = geom1 == self._sat_geom_id
        sat_in_g2 = geom2 == self._sat_geom_id
        sat_mask = sat_in_g1 | sat_in_g2

        # The other geom in each satellite contact (only indexed where sat_mask is true)
        other_geom = np.where(sat_in_g1, geom2, geom1)

        # Build per-node contact index lists
        contact_ci_by_node = {}
        for ci in np.flatnonzero(sat_mask):
            idx = self._geom_id_to_cable_idx.get(int(other_geom[ci]), -1)
            if 0 <= idx < self.active_count:
                if idx in contact_ci_by_node:
                    contact_ci_by_node[idx].append(int(ci))
                else:
                    contact_ci_by_node[idx] = [int(ci)]

        cable_node_indices = sorted(contact_ci_by_node.keys())
        self._cable_sat_contacts_cache = (cable_node_indices, contact_ci_by_node)
        return self._cable_sat_contacts_cache

    def _get_cable_leave_pos(self):
        """Find where the cable departs the satellite surface.

        Returns the position of the highest-index cable node still touching the satellite mesh.
        If the departure node loses contact, a grace period keeps it as the departure point
        for DEPARTURE_GRACE_PERIOD seconds before falling back.
        """

        cable_node_indices = self._find_cable_sat_contacts()[0]

        if cable_node_indices:
            new_leave = cable_node_indices[-1]  # Already sorted ascending
            if new_leave > self._cable_leave_idx:
                # Departure moved toward spool — accept immediately
                self._cable_leave_idx = new_leave
                self._cable_leave_grace_expiry = self.data.time + DEPARTURE_GRACE_PERIOD
            elif new_leave < self._cable_leave_idx and self.data.time >= self._cable_leave_grace_expiry:
                # Departure moved toward anchor and grace period expired — accept the drop
                self._cable_leave_idx = new_leave
                self._cable_leave_grace_expiry = self.data.time + DEPARTURE_GRACE_PERIOD
            return self._get_node_pos(self._cable_leave_idx)

        # No contacts — use the previous departure node if still within the grace period
        if self.data.time < self._cable_leave_grace_expiry:
            self._cable_leave_idx = min(self._cable_leave_idx, self.active_count - 1)
            return self._get_node_pos(self._cable_leave_idx)

        self._cable_leave_idx = 0
        return self.data.site_xpos[self.attachment_site_id].copy()

    def step(self):
        """Advance the simulation by one timestep: apply forces, step physics, and handle cable segment
        spawning/despawning, node freezing, and tension recording."""

        # Apply tension force on spool - must be done before mj_step
        self._apply_tension_and_forces()

        # Step forward
        mujoco.mj_step(self.model, self.data)
        self._cable_sat_contacts_cache = None

        # Check for spawning and despawning nodes
        self._maybe_spawn_despawn()

        # Check for freezing nodes
        if self.data.time >= self._next_freeze_check:
            self._check_frozen_nodes()
            self._next_freeze_check = self.data.time + self._freeze_check_interval

        # Check if tension data should be recorded. This is mainly for debugging.
        if self.data.time >= self._next_tension_record_time:
            self._record_tension()
            self._next_tension_record_time = self.data.time + self._tension_record_interval

    def _check_frozen_nodes(self):
        """Detect cable nodes near the anchor that have stopped moving relative to the satellite."""

        # Get satellite velocity state
        cyl_qvel_adr = self.model.jnt_dofadr[self.cylinder_jnt_id]
        v_sat = self.data.qvel[cyl_qvel_adr:cyl_qvel_adr + 3]
        omega_sat_local = self.data.qvel[cyl_qvel_adr + 3:cyl_qvel_adr + 6]
        R_sat = self.data.xmat[self._phys_cyl_body].reshape(3, 3)
        omega_sat = R_sat @ omega_sat_local  # Now in world coordinate frame
        pos_sat = self.data.xpos[self._phys_cyl_body]

        # Number of pending nodes to freeze is recalculated each time this function is called
        self._pending_freeze_count = 0

        # Only nodes up to the departure point can be frozen — nodes beyond it are still traveling towards the satellite
        n = min(self._cable_leave_idx + 1, self.active_count)
        if n == 0:
            return

        # Vectorized computation of orthogonal relative speed for candidate nodes
        n_fetch = min(n + 1, len(self._phys_cable_body_arr))
        body_ids = self._phys_cable_body_arr[:n_fetch]
        all_pos = self.data.xpos[body_ids]
        pos_cable = all_pos[:n]
        if n_fetch > n:
            pos_next = all_pos[1:]
        else:
            # Last node has no next neighbor — duplicate the last position (direction is irrelevant)
            pos_next = np.empty((n, 3))
            pos_next[:n - 1] = all_pos[1:]
            pos_next[n - 1] = all_pos[n - 1]

        # Cable node velocities via pre-computed DOF addresses
        dof_adrs = self._cable_dof_adrs[:n]
        vel_indices = dof_adrs[:, None] + np.arange(3)
        v_cable = self.data.qvel[vel_indices]

        # Satellite velocity at each cable node: v_sat + omega_sat × (pos - pos_sat)
        v_sat_at_cable = v_sat + np.cross(omega_sat, pos_cable - pos_sat)

        # Relative velocity in the plane orthogonal to the cable direction. We only care about movement
        # in an orthogonal plane because a real cable is continuous, so the location of the nodes along the cable is
        # arbitrary and any component aligned with the cable is just moving the node along the cable!
        v_rel = v_cable - v_sat_at_cable
        cable_dir = pos_next - pos_cable
        cable_lens = np.linalg.norm(cable_dir, axis=1, keepdims=True)
        if np.min(cable_lens) < 1e-9:
            print("WARNING: short cable detected in node freezing calculations.")
        cable_dir /= cable_lens

        # Project out the axial component: v_rel_ortho = v_rel - (v_rel · cable_dir) * cable_dir
        # The np.sum call is a vectorized dot product
        axial = np.sum(v_rel * cable_dir, axis=1, keepdims=True)
        v_rel_ortho = v_rel - axial * cable_dir

        # Read tension at each candidate node for freeze eligibility check. A node can only freeze if its tension has
        # decayed to freeze_max_tension_frac of the cable tension. This ensures nodes are only frozen after the capstan
        # effect has sufficiently reduced their tension, in case this reduction allows the node to move again.
        candidate_tensions = self.cable_stiffness * (self.data.ten_length[self.cable_tendon_ids[:n]] - self.cable_seg_len)
        tension_threshold = self._freeze_max_tension_frac * self.cable_tension

        # Scalar loop for rolling-window freeze detection. The window stores orthogonal velocity vectors, and the
        # freeze criteria is that the magnitude of the mean velocity over the window is below threshold. This filters
        # out high-frequency oscillations (e.g. from contact solver noise) that cancel over time.
        # Additionally, nodes must have tension below the freeze tension threshold.
        mean_vel_magnitudes = np.empty(n)
        for i in range(n):
            # Append velocity vector to rolling window
            if i not in self._freeze_vel_history:
                self._freeze_vel_history[i] = deque(maxlen=self._freeze_window_samples)
            self._freeze_vel_history[i].append(v_rel_ortho[i])

        for i in range(n):
            # Freeze check: magnitude of the mean velocity vector over the window
            history = self._freeze_vel_history[i]
            if len(history) == self._freeze_window_samples:
                mean_vel = np.mean(history, axis=0)
                mean_vel_mag = np.linalg.norm(mean_vel)
            else:
                mean_vel_mag = float('inf')
            mean_vel_magnitudes[i] = mean_vel_mag

            if mean_vel_mag < self._freeze_threshold and candidate_tensions[i] < tension_threshold:
                self._pending_freeze_count += 1
            else:
                break  # Must be contiguous from anchor

    def _record_tension(self):
        """Saves current tendon and free-link tension for plotting.

        Tensions are stored as a dict keyed by absolute node index (anchor_idx + physics_idx) so that each node retains
        the same index across freeze/recompile events. Also records the capstan target tension at each node for
        comparison.
        """
        self.time_history.append(self.data.time)

        tensions = {}
        targets = {}
        contact_indices, _ = self._find_cable_sat_contacts()

        # Compute capstan targets for display: what the capstan equation predicts for each node
        # based on its wrap angle from the departure point (spool side). This uses the active
        # wrap angles WITHOUT the frozen offset — the departure node has target = cable_tension,
        # and tension decays from there toward the anchor.
        contact_set = set(contact_indices)
        if len(contact_indices) >= 2:
            wrap_angles = self._compute_wrap_angles(contact_indices)
            T_target = self.cable_tension * np.exp(-self.capstan_friction * wrap_angles)
            for j, idx in enumerate(contact_indices):
                abs_idx = self.anchor_idx + idx
                targets[abs_idx] = float(T_target[j])

        for i in range(self.active_count):
            abs_idx = self.anchor_idx + i
            if i not in contact_set:
                targets[abs_idx] = self.cable_tension

        for i in range(self.active_count):
            abs_idx = self.anchor_idx + i
            tendon_id = self.cable_tendon_ids[i]
            length = self.data.ten_length[tendon_id]
            tensions[abs_idx] = self.cable_stiffness * (length - self.cable_seg_len)

        tensions['free_link'] = self.cable_tension

        self.tension_history.append(tensions)
        self.target_history.append(targets)

        # Record contact force diagnostics at the same interval
        self._record_contact_diagnostics()

    def _get_contact_friction(self):
        """Extract per-node friction and normal forces for all cable-satellite contacts, projected onto the cable
        direction.

        Uses _find_cable_sat_contacts() for contact identification and then extracts forces only for the identified
        cable-satellite contacts. Friction is projected onto the local cable direction at each node (positive = toward
        the AV).

        Returns:
            node_friction_along_cable: Array of per-node friction projected onto cable direction, indexed by position
                in contact_indices (not by cable_idx).
            node_normal: Array of per-node normal force magnitude, indexed by position in contact_indices.
            cumulative_friction: Sum of all per-node friction along the cable direction.
        """
        contact_indices, contact_ci_by_node = self._find_cable_sat_contacts()

        n_nodes = len(contact_indices)
        if n_nodes == 0:
            return np.empty(0), np.empty(0), 0.0

        ci_arr = np.array(contact_indices, dtype=np.intp)

        # Vectorized cable directions: toward the spool (higher index)
        positions = self.data.xpos[self._phys_cable_body_arr[:self.active_count]]
        at_end = ci_arr >= self.active_count - 1
        cable_dirs = np.where(at_end[:, None],
                              positions[ci_arr] - positions[np.maximum(ci_arr - 1, 0)],
                              positions[np.minimum(ci_arr + 1, self.active_count - 1)] - positions[ci_arr])
        lens = np.linalg.norm(cable_dirs, axis=1, keepdims=True)
        cable_dirs /= lens

        # Flatten all contacts into parallel arrays: (contact_index, owning node position in ci_arr).
        # Pre-compute total size to avoid Python-level appends.
        ncon = self.data.ncon
        counts = np.array([len(contact_ci_by_node[idx]) for idx in contact_indices], dtype=np.intp)
        n_total = counts.sum()
        if n_total == 0:
            return np.zeros(n_nodes), np.zeros(n_nodes), 0.0

        all_ci_arr = np.empty(n_total, dtype=np.intp)
        all_node_pos_arr = np.empty(n_total, dtype=np.intp)
        pos = 0
        for node_pos, cable_idx in enumerate(contact_indices):
            ci_list = contact_ci_by_node[cable_idx]
            n = len(ci_list)
            all_ci_arr[pos:pos + n] = ci_list
            all_node_pos_arr[pos:pos + n] = node_pos
            pos += n

        # Filter out any contacts beyond ncon (shouldn't happen, but guard matches original logic)
        if np.any(all_ci_arr >= ncon):
            valid = all_ci_arr < ncon
            all_ci_arr = all_ci_arr[valid]
            all_node_pos_arr = all_node_pos_arr[valid]
            n_total = len(all_ci_arr)
            if n_total == 0:
                return np.zeros(n_nodes), np.zeros(n_nodes), 0.0

        # Batch extract contact forces
        force_buf = np.empty((n_total, 6))
        for k in range(n_total):
            mujoco.mj_contactForce(self.model, self.data, int(all_ci_arr[k]), force_buf[k])

        # Vectorized sign: +1 if cable is geom1, -1 if geom2
        cable_geom_ids = self._phys_cable_geom_arr[ci_arr[all_node_pos_arr]]
        signs = np.where(self.data.contact.geom1[all_ci_arr] == cable_geom_ids, 1.0, -1.0)

        # Normal forces per contact
        normals_per_contact = signs * force_buf[:, 0]

        # Friction projection: tangent vectors from contact frames
        frames = self.data.contact.frame[all_ci_arr]  # (n_total, 9)
        tangent1 = frames[:, 3:6]
        tangent2 = frames[:, 6:9]
        friction_world = force_buf[:, 1:2] * tangent1 + force_buf[:, 2:3] * tangent2  # (n_total, 3)

        # Project friction onto each contact's node cable direction
        cable_dir_per_contact = cable_dirs[all_node_pos_arr]
        friction_along_per_contact = signs * np.sum(friction_world * cable_dir_per_contact, axis=1)

        # Accumulate per-node
        node_friction_along_cable = np.zeros(n_nodes)
        node_normal = np.zeros(n_nodes)
        np.add.at(node_friction_along_cable, all_node_pos_arr, friction_along_per_contact)
        np.add.at(node_normal, all_node_pos_arr, normals_per_contact)
        cumulative_friction = node_friction_along_cable.sum()

        return node_friction_along_cable, node_normal, cumulative_friction

    def _compute_wrap_angles(self, contact_indices):
        """Compute cumulative wrap angle from the spool-side contact node to each contact node.

        Assumes cylindrical wrapping geometry with the cylinder axis aligned to Z. The wrap angle between consecutive
        contact nodes is the angle between their radial vectors projected onto the XY plane (perpendicular to the
        cylinder axis), so helical pitch does not inflate the wrap angle.

        Args:
            contact_indices: Sorted (ascending) list of cable node indices in contact with the satellite.

        Returns:
            (n,) float array of cumulative wrap angles (radians) aligned with contact_indices, measured from the
            spool-side contact node (whose entry is 0.0).
        """
        if len(contact_indices) < 2:
            return np.zeros(len(contact_indices))

        sat_pos = self.data.xpos[self._phys_cyl_body]
        body_ids = self._phys_cable_body_arr[:self.active_count]
        positions = self.data.xpos[body_ids]

        # Radial vectors projected onto the XY plane (drop Z component)
        ci = np.array(contact_indices)
        r_vecs = positions[ci] - sat_pos
        r_vecs[:, 2] = 0.0
        r_lens = np.linalg.norm(r_vecs, axis=1)

        # Pairwise cosines between consecutive nodes (walking from spool toward anchor)
        # Pairs: (ci[-2], ci[-1]), (ci[-3], ci[-2]), ..., (ci[0], ci[1]) i.e. reversed consecutive pairs
        r_curr = r_vecs[:-1]   # nodes 0..n-2
        r_next = r_vecs[1:]    # nodes 1..n-1
        len_curr = r_lens[:-1]
        len_next = r_lens[1:]

        # Where either radius is degenerate, treat the angle as 0
        valid = (len_curr > 1e-9) & (len_next > 1e-9)
        cos_angles = np.where(valid,
                              np.sum(r_curr * r_next, axis=1) / (len_curr * len_next),
                              1.0)
        np.clip(cos_angles, -1.0, 1.0, out=cos_angles)
        deltas = np.arccos(cos_angles)

        # Cumulative wrap from spool side: reverse, cumsum, reverse back
        cumulative = np.cumsum(deltas[::-1])[::-1]

        # Spool-side node (last) has wrap angle 0
        return np.append(cumulative, 0.0)

    def _record_contact_diagnostics(self):
        """Record contact force diagnostics for cable-satellite contacts."""
        contact_indices, contact_ci_by_node = self._find_cable_sat_contacts()
        node_friction_along_cable, node_normal, cumulative_friction = self._get_contact_friction()

        # Use the force snapshot taken at the end of _apply_tension_and_forces, where all three
        # arrays (constraint, passive, applied) are from the same consistent moment.
        snap = self._sat_force_snapshot
        sat_f_constraint = snap['constraint']
        sat_f_passive = snap['passive']
        sat_f_applied = snap['applied']

        self.contact_diag_history.append({
            'num_contacts': len(contact_indices),
            'total_normal': node_normal.sum() if len(node_normal) else 0.0,
            'total_friction': np.abs(node_friction_along_cable).sum() if len(node_friction_along_cable) else 0.0,
            'avg_utilization': (
                np.mean(np.abs(node_friction_along_cable[node_normal > 1e-9])
                        / (self.cable_friction[0] * node_normal[node_normal > 1e-9]))
                if len(node_normal) and np.any(node_normal > 1e-9) else 0.0
            ),
            'cumulative_friction_along_cable': cumulative_friction,
            'sat_f_constraint': np.linalg.norm(sat_f_constraint),
            'sat_f_passive': np.linalg.norm(sat_f_passive),
            'sat_f_applied': np.linalg.norm(sat_f_applied),
            'sat_f_total': np.linalg.norm(sat_f_constraint + sat_f_passive + sat_f_applied),
            'sat_f_deficit': np.linalg.norm(snap['F_target'] - (sat_f_constraint + sat_f_passive + sat_f_applied)),
        })

    def _apply_tension_and_forces(self):
        """Compute and apply all external forces for the current timestep: cable tension on the spool node,
        AV thruster reaction force, and lateral damping on affected cable nodes."""
        av_pos = self._get_av_pos()
        torque_vec = self._torque_vec  # Pre-allocated, always zero
        qfrc_applied = self._qfrc_buf
        qfrc_applied[:] = 0

        # Apply full tension at the spool node only
        spool_pos = self.data.xpos[self._phys_cable_body_arr[self.spool_idx]]
        toward_av = av_pos - spool_pos
        dist = sqrt(toward_av @ toward_av)

        # This should never happen since the node should despawn first
        if dist < 1e-9:
            print("WARNING: short cable detected in tension application calculations.")

        direction = toward_av / dist
        force = direction * self.cable_tension
        body_id = self._phys_cable_body_arr[self.spool_idx]
        mujoco.mj_applyFT(self.model, self.data, force, torque_vec, spool_pos, body_id, qfrc_applied)

        # Reaction force on the AV
        self.current_thruster_force = self.thruster_react_func(-1 * direction * self.cable_tension)
        mujoco.mj_applyFT(self.model, self.data, self.current_thruster_force, torque_vec, av_pos, self.av_body_id,
                          qfrc_applied)

        # Equal and opposite force on the satellite: AV rigidly tracks satellite translation. The factor of 0.95
        # allows the satellite to slowly move towards the AV. This is the physical equivalent of the AV thrusters
        # not quite being able to keep it away from the satellite
        sat_pos = self.data.xpos[self._phys_cyl_body]
        mujoco.mj_applyFT(self.model, self.data, -self.current_thruster_force * 0.95, torque_vec,
                          sat_pos, self._phys_cyl_body, qfrc_applied)

        # Capstan friction correction: apply supplementary friction forces to cable nodes in contact with the
        # satellite. MuJoCo's discrete contact friction underestimates the continuous capstan effect
        # (T_high / T_low = e^(mu * theta)). The correction computes the expected friction at each contact node
        # from the capstan equation and applies the deficit as an external force on the cable node only (no
        # satellite reaction), since MuJoCo's contact solver already applies a (partial) reaction to the satellite.
        # The correction is applied every step but only recomputed at capstan_recompute_dt intervals.
        if CAPSTAN_CORRECTION:
            if self.data.time >= self._next_capstan_recompute:
                self._recompute_capstan_correction(qfrc_applied, torque_vec)
                self._next_capstan_recompute = self.data.time + self.capstan_recompute_dt
            else:
                # Replay the cached correction from the last recompute
                qfrc_applied += self._capstan_qfrc_delta

        # Vectorized lateral damping: applied to nodes from leave_idx + UNDAMPED_LATERAL_NODES_BEFORE_SAT through
        # spool_idx. Damps velocity orthogonal to the local tendon direction. MuJoCo handles damping in the tendon along
        # the cable direction. This code needs to be vectorized to keep computation time reasonable.
        damping_start = min(self._cable_leave_idx + UNDAMPED_LATERAL_NODES_BEFORE_SAT, self.spool_idx)
        n_damped = self.spool_idx - damping_start + 1

        if n_damped > 0:
            # Read all active positions in one shot
            damping_end = self.spool_idx + 1  # Exclusive

            body_ids = self._phys_cable_body_arr[:damping_end]
            all_pos = self.data.xpos[body_ids]

            # Compute tendon directions from neighbors for damped nodes
            # For interior nodes: dir = pos[i+1] - pos[i-1]
            # For first damped (if d_start==0): dir = pos[1] - pos[0]
            # For last damped (spool_idx): dir = pos[i] - pos[i-1]
            if damping_start > 0:
                prev_pos = all_pos[damping_start - 1:damping_end - 1]  # i-1 for each damped node
            else:
                # First node uses forward difference
                prev_pos = np.empty((n_damped, 3))
                prev_pos[0] = all_pos[0]
                prev_pos[1:] = all_pos[damping_start:damping_end - 1]

            if damping_end <= self.spool_idx:
                next_pos = all_pos[damping_start + 1:damping_end + 1]  # i+1 for each damped node
            else:
                # Last node uses backward difference
                next_pos = np.empty((n_damped, 3))
                next_pos[:-1] = all_pos[damping_start + 1:damping_end]
                next_pos[-1] = all_pos[self.spool_idx]

            tendon_dirs = next_pos - prev_pos  # (n_damped, 3)
            tendon_lens = np.linalg.norm(tendon_dirs, axis=1, keepdims=True)

            if np.min(tendon_lens) < 1e-9:
                print("WARNING: short cable detected in damping calculations.")

            tendon_dirs /= tendon_lens

            # Read velocities for damped nodes via fancy indexing
            # This hard to read code is necessary to vectorize...
            dof_adrs = self._cable_dof_adrs[damping_start:damping_end]
            vel_indices = dof_adrs[:, None] + np.arange(3)  # (n_damped, 3) index array
            vels = self.data.qvel[vel_indices]  # (n_damped, 3)

            # Compute lateral velocity: v - (v·d)*d
            axial_proj = np.sum(vels * tendon_dirs, axis=1, keepdims=True)
            lateral_vels = vels - axial_proj * tendon_dirs

            # Damping forces
            damp_forces = -self.cable_lateral_damping * lateral_vels  # (n_damped, 3)

            # Write forces directly into qfrc_applied using the same index array
            np.add.at(qfrc_applied, vel_indices, damp_forces)

            # Store damping range for color updates at render time
            self._damping_start = damping_start

        self.data.qfrc_applied[:] = qfrc_applied

        # Snapshot the satellite forces for diagnostic recording. Done here (after all forces are written to
        # qfrc_applied but before mj_step) so all three arrays are from the same moment:
        # - qfrc_constraint/qfrc_passive: from the previous mj_step (latest available)
        # - qfrc_applied: the buffer we just finished building for this step
        # Only translational forces (DOFs 0-2, world frame) are stored for the force plot.
        sat_dof = self.model.jnt_dofadr[self.cylinder_jnt_id]
        self._sat_force_snapshot = {
            'constraint': self.data.qfrc_constraint[sat_dof:sat_dof + 3].copy(),
            'passive': self.data.qfrc_passive[sat_dof:sat_dof + 3].copy(),
            'applied': self.data.qfrc_applied[sat_dof:sat_dof + 3].copy(),
            'F_target': force.copy(),  # spool tension vector (toward AV)
        }

    def _recompute_capstan_correction(self, qfrc_applied, torque_vec):
        """Recompute capstan friction correction forces and cache the qfrc delta for replay on subsequent steps."""
        self._capstan_qfrc_delta[:] = 0

        contact_indices, _ = self._find_cable_sat_contacts()
        node_friction_along_cable, node_normal, _ = self._get_contact_friction()

        if len(contact_indices) < 2:
            return

        wrap_angles = self._compute_wrap_angles(contact_indices)

        # The active wrap angles start from 0 at the spool side, but the frozen chain adds additional wrap
        # before the first active node. Add the total frozen wrap so that T_target correctly reflects the full
        # cable path.
        wrap_angles += self._frozen_wrap_total

        ci = np.array(contact_indices)

        # Feedforward capstan correction: apply the predicted capstan friction force at each contact node. The
        # capstan equation gives the target tension profile, and the friction at each node is the tension drop
        # across it. Apply the full predicted friction without subtracting MuJoCo's contribution (since MuJoCo's
        # velocity-dependent friction competes with applied forces rather than adding to them).
        # Equal-and-opposite reactions are applied to the satellite (Newton's third law).
        T_target = self.cable_tension * np.exp(-self.capstan_friction * wrap_angles)

        # Friction force at each node: the force needed to hold this node at its target tension given the actual
        # tension on the spool side. Using the actual spool-side tension (not the target) correctly accounts for
        # unconverged neighbors — if the spool-side neighbor is still at 10N but this node's target is 3N, the
        # friction needed is ~7N, not the steady-state 0.5N that the target-based calculation gives. This reads
        # the neighbor's tension from MuJoCo's springs, not from the corrections, so it doesn't create a
        # feedback loop on the node's own tension.
        actual_tensions = self.cable_stiffness * (self.data.ten_length[self.cable_tendon_ids[ci]] - self.cable_seg_len)

        T_spool_actual = np.append(actual_tensions[1:], self.cable_tension)
        # Friction is only applied when the spool-side tension exceeds the target, meaning MuJoCo underestimates
        # the friction. If MuJoCo overestimates the friction, apply no correction (here the 0N)
        capstan_friction_forces = np.maximum(0, (T_spool_actual - T_target))

        # Clamp supplement at each node to not exceed mu * N (the physical friction limit). This prevents the
        # correction from pushing low-normal-force nodes off the satellite surface.
        # NOTE: The limit to the external correction tension that can be applied is based on the MujoCo friction since
        # this is the largest frictional force MuJoCo can apply, and MuJoCo must be able to oppose the applied force if
        # to ensure the simulation remains stable. Better convergence requires increasing self.cable_friction[0], which
        # also has the tendency to fix nodes in place sooner and reduce sliding.
        np.minimum(capstan_friction_forces, self.cable_friction[0] * node_normal, out=capstan_friction_forces)


        # Cable directions for all contact nodes (toward anchor / lower index)
        body_ids_all = self._phys_cable_body_arr[:self.active_count]
        positions = self.data.xpos[body_ids_all]
        next_pos = np.where((ci < self.active_count - 1)[:, None],
                            positions[np.minimum(ci + 1, self.active_count - 1)],
                            positions[ci])
        prev_pos = np.where((ci > 0)[:, None],
                            positions[np.maximum(ci - 1, 0)],
                            positions[ci])
        cable_dirs = prev_pos - next_pos  # toward anchor
        cable_dirs /= np.linalg.norm(cable_dirs, axis=1, keepdims=True)

        # Apply supplement forces where required. Note this correction only helps when friction is
        # underestimated, meaning that capstan_friction_forces > 0.
        mask = capstan_friction_forces > 0
        if np.any(mask):
            ci_masked = ci[mask]
            dirs_masked = cable_dirs[mask]
            supp_masked = capstan_friction_forces[mask]

            correction_forces = dirs_masked * supp_masked[:, None]
            dof_adrs = self._cable_dof_adrs[ci_masked]
            dof_indices = dof_adrs[:, None] + np.arange(3)
            np.add.at(self._capstan_qfrc_delta, dof_indices, correction_forces)

            # Equal-and-opposite reaction on the satellite (Newton's third law)
            sat_body = self._phys_cyl_body
            for k in range(len(ci_masked)):
                node_pos = positions[ci_masked[k]]
                reaction = -correction_forces[k]
                mujoco.mj_applyFT(self.model, self.data, reaction, torque_vec,
                                  node_pos, sat_body, self._capstan_qfrc_delta)

        qfrc_applied += self._capstan_qfrc_delta

    def _get_av_pos(self) -> np.ndarray:
        # TODO: note this effectively fixes the AV in place, and must be changed to test different control schemes
        """Return the current position of the Autonomous Vehicle (AV) as a numpy array."""
        return np.array((self.av_init_pos[0], self.av_init_pos[1], self.av_init_pos[2]))

    def _world_to_sat_body_frame(self, world_pos):
        """Convert a world-frame position to the satellite's body frame.

        Args:
            world_pos: 3-element array-like position in world coordinates.
        """
        # Compute inverse rotation: negate the quaternion
        sat_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(sat_quat_inv, self.data.xquat[self._phys_cyl_body])
        # Rotate (world_pos - sat_pos) by inverse quaternion
        delta = np.array(world_pos, dtype=np.float64) - self.data.xpos[self._phys_cyl_body]
        body_frame_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(body_frame_pos, delta, sat_quat_inv)
        return body_frame_pos

    def _freeze_anchor_nodes(self, freeze_count: int):
        """Freeze `freeze_count` nodes from the anchor end: store body-frame positions and
        prepare for recompilation that drops frozen nodes from the physics model.

        Args:
            freeze_count: Number of contiguous nodes starting from the anchor end to freeze.
        """
        if freeze_count <= 0:
            return

        # Store body-frame positions and inter-node wrap angles for nodes being frozen. The inter-node angle at each
        # node is the angle between radial vectors from the satellite COM to adjacent nodes — the same geometry used by
        # _compute_wrap_angles.
        sat_pos = self.data.xpos[self._phys_cyl_body]

        # Collect world positions of: previous anchor (last frozen or original attach), nodes being frozen, and next node
        prev_anchor_world = np.zeros(3)
        if self._frozen_body_frame_pos:
            sat_quat = self.data.xquat[self._phys_cyl_body]
            mujoco.mju_rotVecQuat(prev_anchor_world, self._frozen_body_frame_pos[-1], sat_quat)
            prev_anchor_world += sat_pos
        else:
            sat_quat = self.data.xquat[self._phys_cyl_body]
            mujoco.mju_rotVecQuat(prev_anchor_world, self.sat_attach_pos, sat_quat)
            prev_anchor_world += sat_pos

        freeze_world_positions = [prev_anchor_world]
        for i in range(freeze_count):
            freeze_world_positions.append(self._get_node_pos(i))

        # Compute inter-node angles and store body-frame positions
        for i in range(freeze_count):
            world_pos = freeze_world_positions[i + 1]
            body_pos = self._world_to_sat_body_frame(world_pos)
            self._frozen_body_frame_pos.append(body_pos)

            # Angle between radial vectors from satellite COM
            r_prev = freeze_world_positions[i] - sat_pos
            r_curr = freeze_world_positions[i + 1] - sat_pos
            len_prev = np.linalg.norm(r_prev)
            len_curr = np.linalg.norm(r_curr)
            if len_prev < 1e-9 or len_curr < 1e-9:
                print("WARNING! Error with length calculations in _freeze_anchor_nodes")

            # np.clip call here protects against round-off error causing a value just outside the domain of np.arccos,
            # which would lead to NaN and break the simulation when clipping is a much better approximation
            cos_angle = np.clip(np.dot(r_prev, r_curr) / (len_prev * len_curr), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            self._frozen_inter_angles.append(angle)
            self._frozen_wrap_total += float(angle)

        # Place the new anchor so the anchor tendon preserves the actual tension at the first
        # non-frozen node. Using seg_equilibrium_len (based on full cable tension) would reset
        # the anchor tendon to T_cable, undoing the capstan tension decay.
        new_cable0_world_pos = self._get_node_pos(freeze_count)
        last_frozen_world_pos = self._get_node_pos(freeze_count - 1)
        cable_dir = new_cable0_world_pos - last_frozen_world_pos
        cable_dir_len = np.linalg.norm(cable_dir)
        if cable_dir_len < 1e-9:
            print("WARNING: short cable detected when freezing the anchor node!")

        cable_dir /= cable_dir_len

        # Use the actual tendon length at the first non-frozen node as the offset, so the
        # anchor tendon in the recompiled model starts at the same tension
        tid = self.cable_tendon_ids[freeze_count]
        actual_tendon_len = self.data.ten_length[tid]
        anchor_world_pos = new_cable0_world_pos - cable_dir * actual_tendon_len
        self._current_anchor_body_pos = self._world_to_sat_body_frame(anchor_world_pos)

        # Update indices
        self.anchor_idx += freeze_count
        self.spool_idx -= freeze_count
        self.active_count -= freeze_count

        # Shift freeze tracking indices to match new physics model
        self._freeze_vel_history = {k - freeze_count: v
                                    for k, v in self._freeze_vel_history.items() if k >= freeze_count}
        self._cable_leave_idx = max(0, self._cable_leave_idx - freeze_count)
        self._pending_freeze_count = 0

        print(f"  [FREEZE] Froze {freeze_count} nodes, anchor_idx now {self.anchor_idx}, active: {self.active_count}")

    def _maybe_spawn_despawn(self):
        """Check if the free link between the spool and AV has grown long enough to warrant spawning a new cable
        segment or shrunk far enough that one cable segment must be removed. Trigger the correct action if so."""
        if self.active_count < self.max_seg_num:
            av_pos = self._get_av_pos()
            d = av_pos - self._get_node_pos(self.spool_idx)
            free_link_length = sqrt(d @ d)

            # Calculate target free link length and see if another node is required
            cable_leave_pos = self._get_cable_leave_pos()
            d = av_pos - cable_leave_pos
            target_free_link = sqrt(d @ d) * FREE_LINK_AV_SAT_DIST_RATIO

            # Spawn only when free link has grown more than some threshold
            # This prevents nodes from repeatedly spawning and despawning due to small oscillations, which can happen
            # when the spawn and despawn positions are the same
            if free_link_length > target_free_link + self.seg_equilibrium_len * SPAWN_NEW_NODE_GROWTH_THRESHOLD:
                print(f"\n=== SPAWN TRIGGER t={self.data.time:.4f}s ===")
                print(f"Free link: {free_link_length:.4f}m > target {target_free_link:.4f}m + {self.seg_equilibrium_len * 1.1:.4f}m")
                self._spawn_segment()

            # Despawn only when the free link is too short, controlled by a threshold
            elif free_link_length < target_free_link - self.seg_equilibrium_len * DESPAWN_NODE_SHRINK_THRESHOLD:
                print(f"\n=== DESPAWN TRIGGER t={self.data.time:.4f}s ===")
                print(f"Free link: {free_link_length:.4f}m < target {target_free_link:.4f}m - {self.seg_equilibrium_len * 0.1:.4f}m")
                self._despawn_segment()

    def _spawn_segment(self):
        """Activate the next cable segment: position it at equilibrium distance from the current spool toward the AV,
        transfer velocity, enable the connecting tendon, and update indices. Triggers a model recompilation if all
        compiled segments are in use. Note that freezing nodes alone cannot trigger a recompile—it must occur due to
        node spawning."""
        # Forward the physics state. This must be done at the start AND at the end to get the proper location for the
        # old spool
        mujoco.mj_forward(self.model, self.data)

        # Check if a recompilation is necessary. If recompiling, also drop the frozen nodes from the physics simulation
        if ENABLE_RECOMPILE and self.active_count >= self.compiled_seg_count:
            frozen_now = 0
            if self._pending_freeze_count > 0:
                frozen_now = self._pending_freeze_count
                self._freeze_anchor_nodes(frozen_now)

            new_count = min(self.active_count + RECOMPILE_BUFFER, self.max_seg_num)
            self._recompile_model(new_count, anchor_body_frame_pos=self._current_anchor_body_pos,
                                  frozen_this_recompile=frozen_now)
            mujoco.mj_forward(self.model, self.data)

        # Get old spool data from the MuJoCo model
        old_spool_pos = self._get_node_pos(self.spool_idx)
        old_joint_id = self.cable_jnt_ids[self.spool_idx]
        old_qvel_adr = self.model.jnt_dofadr[old_joint_id]

        # Get new segment's joint addresses
        qpos_adr = self.model.jnt_qposadr[self.cable_jnt_ids[self.active_count]]
        qvel_adr = self.model.jnt_dofadr[self.cable_jnt_ids[self.active_count]]

        # Direction from old spool toward origin
        toward_origin = self._get_av_pos() - old_spool_pos
        toward_origin_dir = toward_origin / np.linalg.norm(toward_origin)

        # Position new spool at equilibrium distance from old spool, toward origin
        spawn_pos = old_spool_pos + toward_origin_dir * self.seg_equilibrium_len

        # Set new segment position and orientation
        self.data.qpos[qpos_adr:qpos_adr+3] = spawn_pos
        self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

        # To result in smoother spawning, give the cable an initial velocity
        self.data.qvel[qvel_adr:qvel_adr+3] = self.data.qvel[old_qvel_adr:old_qvel_adr+3]

        # Activate tendon arriving at the new segment (connects old spool to new node)
        tendon_id = self.cable_tendon_ids[self.active_count]
        self.model.tendon_stiffness[tendon_id] = self.cable_stiffness
        self.model.tendon_damping[tendon_id] = self.cable_axial_damping

        # Update spool index and active count
        self.spool_idx = self.active_count
        self.active_count += 1

        # Now that the new spool is spawned, forward the state again. Forwarding twice is required to reduce tension
        # spikes
        mujoco.mj_forward(self.model, self.data)

    def _despawn_segment(self):
        """Deactivate the current spool segment: disable its tendon and contacts, move it behind the AV, promote the
        previous segment to spool, and recompile if too many DOFs are unused."""
        # Forward the physics state first
        mujoco.mj_forward(self.model, self.data)

        old_spool = self.spool_idx
        new_spool = old_spool - 1  # Previous segment becomes the new spool

        print(f"DESPAWN: old_spool={old_spool}, new_spool={new_spool}")

        # Deactivate the tendon arriving at old_spool (connects new_spool to old_spool)
        tendon_id = self.cable_tendon_ids[old_spool]
        self.model.tendon_stiffness[tendon_id] = 0
        self.model.tendon_damping[tendon_id] = 0

        # Move the old spool segment far away and zero its velocity
        old_spool_jnt_id = self.cable_jnt_ids[old_spool]
        qpos_adr = self.model.jnt_qposadr[old_spool_jnt_id]
        qvel_adr = self.model.jnt_dofadr[old_spool_jnt_id]

        # Place it behind the AV, one segment length per inactive body
        inactive_pos = self._get_av_pos() + (old_spool - self.active_count + 1) * self.seg_equilibrium_len

        self.data.qpos[qpos_adr:qpos_adr + 3] = inactive_pos
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0

        # Update spool index and active count
        self.spool_idx = new_spool
        self.active_count = old_spool  # active_count is now old_spool (not old_spool + 1)

        print(f"  new spool_idx: {self.spool_idx}, active_count: {self.active_count}")

        # Recompile to shed excess DOFs if we have too many unused segments
        if ENABLE_RECOMPILE and self.compiled_seg_count - self.active_count > 2 * RECOMPILE_BUFFER:
            new_count = self.active_count + RECOMPILE_BUFFER
            self._recompile_model(new_count)

        # Allow the leave index to drop by 1 since the spool retracted
        if self._cable_leave_idx > 0:
            self._cable_leave_idx -= 1

        # Forward again to update state
        mujoco.mj_forward(self.model, self.data)

    def plot_tension(self):
        """Plot the tension history of cable links over time, using absolute node indices so each
        node maintains its identity across freeze/recompile events. Mainly intended for debugging."""
        plt.figure(figsize=(14, 8), dpi=250)

        # Collect all absolute indices that ever appeared
        all_abs_indices = set()
        for tensions in self.tension_history:
            all_abs_indices.update(k for k in tensions if k != 'free_link')
        all_abs_indices = sorted(all_abs_indices)

        # Use tab20 colormap cycling every 20 nodes so colors are consistent across freezes
        cmap = plt.colormaps['tab20']

        # Plot each absolute link (solid) and its capstan target (dashed, same color)
        for abs_idx in all_abs_indices:
            color = cmap((abs_idx % 20) / 20)

            times = []
            data = []
            for i, tensions in enumerate(self.tension_history):
                if abs_idx in tensions:
                    times.append(self.time_history[i])
                    data.append(tensions[abs_idx])
            if times:
                plt.plot(times, data, color=color,
                         label=f'Link {abs_idx}' if abs_idx < 20 else '', alpha=0.7)

            if self.target_history:
                t_times = []
                t_data = []
                for i, targets in enumerate(self.target_history):
                    if abs_idx in targets:
                        t_times.append(self.time_history[i])
                        t_data.append(targets[abs_idx])
                if t_times:
                    plt.plot(t_times, t_data, linestyle='--', color=color, alpha=0.4, linewidth=0.8)

        # Free link (spool) tension
        spool_times = []
        spool_data = []
        for i, tensions in enumerate(self.tension_history):
            if 'free_link' in tensions:
                spool_times.append(self.time_history[i])
                spool_data.append(tensions['free_link'])
        if spool_times:
            plt.plot(spool_times, spool_data, label='Free link tension', linewidth=2, color='red')

        plt.axhline(y=self.cable_tension, color='gray', linestyle='--', label='Applied tension')

        plt.xlabel('Time (s)')
        plt.ylabel('Tension (N)')
        plt.title('Link Tension Over Time (absolute node index)')
        plt.ylim(bottom=0)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Error plot: actual tension - target tension (positive = above target, negative = below)
        if self.target_history:
            plt.figure(figsize=(14, 6), dpi=500)
            for abs_idx in all_abs_indices:
                color = cmap((abs_idx % 20) / 20)
                e_times = []
                e_data = []
                for i in range(len(self.tension_history)):
                    if abs_idx in self.tension_history[i] and i < len(self.target_history) and abs_idx in self.target_history[i]:
                        e_times.append(self.time_history[i])
                        e_data.append(self.tension_history[i][abs_idx] - self.target_history[i][abs_idx])
                if e_times:
                    plt.plot(e_times, e_data, color=color,
                             label=f'Link {abs_idx}' if abs_idx < 20 else '', alpha=0.7)

            # Average tension error across all active nodes at each timestep
            avg_times = []
            avg_errors = []
            for i in range(len(self.tension_history)):
                errors_at_t = []
                for abs_idx in all_abs_indices:
                    if abs_idx in self.tension_history[i] and i < len(self.target_history) and abs_idx in self.target_history[i]:
                        errors_at_t.append(self.tension_history[i][abs_idx] - self.target_history[i][abs_idx])
                if errors_at_t:
                    avg_times.append(self.time_history[i])
                    avg_errors.append(np.mean(errors_at_t))
            if avg_times:
                plt.plot(avg_times, avg_errors, color='black', linewidth=2, label='Average error')

            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.xlabel('Time (s)')
            plt.ylabel('Tension Error (N)')
            plt.title('Tension Error (actual - target) Over Time')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    def plot_satellite_forces(self):
        """Plot forces on the satellite over time, read directly from MuJoCo's per-DOF force arrays.
        Shows constraint forces (contacts), passive forces (tendon springs/damping), applied forces
        (user-applied external), and their vector sum. The total should approximately match the cable
        tension magnitude in quasi-static conditions."""

        times = self.time_history[:len(self.contact_diag_history)]
        f_constraint = [s['sat_f_constraint'] for s in self.contact_diag_history]
        f_passive = [s['sat_f_passive'] for s in self.contact_diag_history]
        f_applied = [s['sat_f_applied'] for s in self.contact_diag_history]
        f_total = [s['sat_f_total'] for s in self.contact_diag_history]
        f_deficit = [s['sat_f_deficit'] for s in self.contact_diag_history]

        fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
        ax.plot(times, f_constraint, label='Constraint (contacts)', color='tab:blue', alpha=0.8)
        ax.plot(times, f_passive, label='Passive (tendon springs/damping)', color='tab:orange', alpha=0.8)
        ax.plot(times, f_applied, label='Applied (capstan reactions)', color='tab:green', alpha=0.8)
        ax.plot(times, f_total, label='Total (vector sum)', color='tab:red', linewidth=1.5)
        ax.plot(times, f_deficit, label='Remaining deficit to match tension', color='tab:purple',
                linestyle=':', alpha=0.8)
        ax.axhline(y=self.cable_tension, color='gray', linestyle='--', alpha=0.5, label='Cable tension')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force magnitude (N)')
        ax.set_title('Forces on Satellite from Cable')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def render_overlay(self, scene, viewer):
        """Render simulation parameters and live stats as floating 3D text labels.

        Computes label positions dynamically from the current camera state so they always appear on the right side of
        the screen, even if the user orbits, zooms, or resizes the window.

        Args:
            scene: MuJoCo mjvScene to append label geoms to.
            viewer: Active MuJoCo passive viewer, used to read camera state and viewport size.
        """
        # Normalized rotation axis for display
        rot_norm = self.sat_rotation_axis / np.linalg.norm(self.sat_rotation_axis)
        sat_vel = self.data.qvel[self.model.jnt_dofadr[self.cylinder_jnt_id] + 3:
                                 self.model.jnt_dofadr[self.cylinder_jnt_id] + 6]

        lines = [
            "--- Simulation Parameters ---",
            f"Rotation Axis: ({rot_norm[0]:.2f}, {rot_norm[1]:.2f}, {rot_norm[2]:.2f})",
            f"Cable Tension: {self.cable_tension:.1f} N",
            f"AV Pos: ({self.av_init_pos[0]:.2f}, {self.av_init_pos[1]:.2f}, {self.av_init_pos[2]:.2f})",
            f"Attach Pos: ({self.sat_attach_pos[0]:.2f}, {self.sat_attach_pos[1]:.2f}, {self.sat_attach_pos[2]:.2f})",
            f"Target Mass: {self.sat_mass:.0f} kg",
            #             f"AV Mass: {self.av_mass:.0f} kg",
            "",
            #             "--- Live Stats ---",
            f"Time: {self.data.time:.1f} s",
            f"Initial Angular Vel: {self.sat_omega:.2f} rad/s",
            f"Current Angular Vel: {np.linalg.norm(sat_vel):.2f} rad/s",
            #             f"Thruster Force: ({self.current_thruster_force[0]:.1f}, {self.current_thruster_force[1]:.1f},"
            #             + f"{self.current_thruster_force[2]:.1f})",
        ]

        # --- Compute camera basis vectors in world coordinates ---
        # MuJoCo convention: azimuth=0 → looking along +x, azimuth=90 → looking along +y
        # elevation > 0 → looking up, elevation < 0 → looking down
        az = np.radians(viewer.cam.azimuth)
        el = np.radians(viewer.cam.elevation)
        dist = viewer.cam.distance
        lookat = np.array(viewer.cam.lookat, dtype=np.float64)

        # Forward (camera viewing) direction: the direction the camera looks toward
        forward = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ])
        forward /= np.linalg.norm(forward)

        # Screen-right = forward × world_up (MuJoCo uses z-up)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-6:
            right = np.array([-np.sin(az), np.cos(az), 0.0])
        else:
            right /= r_norm

        # Screen-up = right × forward
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        # Compute the visible extents at the lookat plane
        fovy = np.radians(self.display_model.vis.global_.fovy)
        half_height = dist * np.tan(fovy / 2.0)

        # Try to get actual viewport aspect ratio from the viewer; fall back to 16:9
        try:
            vp = viewer.viewport
            aspect = vp.width / max(vp.height, 1)
        except (AttributeError, TypeError):
            aspect = 16.0 / 9.0

        half_width = half_height * aspect

        # Place labels on the right edge, upper portion of the view
        # Text labels are left-anchored at the geom position, so offset slightly inward
        base_pos = lookat + right * half_width * 0.65 + up * half_height * 0.75
        line_spacing = half_height * 0.065  # Scale spacing with zoom level

        for i, text in enumerate(lines):
            if not text:
                continue  # skip blank spacer lines
            if scene.ngeom >= scene.maxgeom - 1:
                break

            pos = base_pos - up * (i * line_spacing)

            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                5 * np.array([0.001, 0, 0], dtype=np.float64),
                pos,
                np.eye(3, dtype=np.float64).flatten(),
                np.array([0, 0, 0, 0], dtype=np.float32),
            )
            g.label = text
            scene.ngeom += 1

    def run(self):
        with mujoco.viewer.launch_passive(self.display_model, self.display_data) as viewer:
            viewer.cam.azimuth = -0.8289683948863515
            viewer.cam.elevation = -21.25310724431816
            viewer.cam.distance = 34.68901593838663
            viewer.cam.lookat[:] = [0, 5, 0]

            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
            time.sleep(5)  # Pause to allow for looking at the scene before the simulation starts and making sure it looks good

            render_interval = 1.0 / 60.0  # Sync display at ~60 Hz real time
            last_render_wall = time.monotonic()
            wall_start = last_render_wall

            # Recording setup. Quartz, imageio, and av are only dependencies when RECORD = True.
            # Uses macOS Quartz CoreGraphics to capture at full Retina resolution.
            # The Quartz capture itself blocks the main thread (for timing accuracy), but all subsequent
            # processing (pixel data extraction, numpy conversion, H.264 encoding) runs on a background thread.
            record_thread = None
            if RECORD:
                import Quartz.CoreGraphics as CG
                import imageio.v3 as iio
                from threading import Thread, Event

                record_dt = 1.0 / RECORD_FPS
                next_record_time = 0.0
                frame_count = 0
                capture_event = Event()   # Main thread signals "capture now"
                done_event = Event()      # Worker signals "capture taken, you can resume"
                stop_event = Event()      # Main thread signals "shut down"
                _shared_cg_image = [None]  # Mutable container to pass CG image between threads

                def _record_worker():
                    """Background thread: captures screen, converts, and encodes to video."""
                    writer = iio.imopen(self.record_output, "w", plugin="pyav")
                    writer.init_video_stream("libx264", fps=RECORD_FPS)
                    while True:
                        capture_event.wait()
                        capture_event.clear()
                        if stop_event.is_set():
                            break
                        # Capture screenshot (fast) then unblock main thread
                        _shared_cg_image[0] = CG.CGWindowListCreateImage(
                            CG.CGRectInfinite, CG.kCGWindowListOptionOnScreenOnly,
                            CG.kCGNullWindowID, CG.kCGWindowImageDefault)
                        done_event.set()
                        # Heavy processing — main thread is already running again
                        cg_image = _shared_cg_image[0]
                        w = CG.CGImageGetWidth(cg_image)
                        h = CG.CGImageGetHeight(cg_image)
                        raw = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(cg_image))
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
                        # For entire monitor or other computers, use frame = frame[:, :, 2::-1].copy()  # BGRA → RGB
                        frame = frame[122:, 540:2916, 2::-1].copy()  # BGRA → RGB
                        frame = frame[:h - h % 2, :w - w % 2]  # libx264 requires even dimensions
                        writer.write_frame(frame)
                    writer.close()

                record_thread = Thread(target=_record_worker, daemon=True)
                record_thread.start()
                print(f"Recording enabled: {RECORD_FPS} fps (sim time) → {self.record_output}")

            # Track sim time vs wall time for performance plotting
            perf_sim_times = []
            perf_wall_times = []
            perf_record_interval = 0.1  # Record every 0.1s sim time
            next_perf_record = 0.0

            try:
                while viewer.is_running():
                    # Run physics steps until wall clock says it's time to render
                    now = time.monotonic()
                    while now - last_render_wall < render_interval:
                        self.step()
                        now = time.monotonic()

                        # Record performance data at fixed sim-time intervals
                        # This is used to benchmark simulation speed and confirm the effectiveness of optimizations
                        if self.data.time >= next_perf_record:
                            perf_sim_times.append(self.data.time)
                            perf_wall_times.append(now - wall_start)
                            next_perf_record = self.data.time + perf_record_interval

                    last_render_wall = now

                    # Sync display once per render frame
                    node_positions, node_colors = self.sync_display()

                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        self.render_cable(viewer.user_scn, node_positions, node_colors)
                        self.render_overlay(viewer.user_scn, viewer)

                    viewer.sync()

                    # Trigger screen capture on background thread, block until capture is taken
                    if RECORD and self.data.time >= next_record_time:
                        capture_event.set()
                        done_event.wait()
                        done_event.clear()
                        frame_count += 1
                        next_record_time += record_dt

                    # Plot tension
                    if self.data.time > self.end_simulation_time:
                        self.plot_tension()
                        self.plot_satellite_forces()
                        self.plotted = True

                        self.perf_plotted = True
                        plt.figure(figsize=(10, 6))
                        plt.plot(perf_wall_times, perf_sim_times, linewidth=2)
                        plt.xlabel('Wall Clock Time (s)')
                        plt.ylabel('Simulation Time (s)')
                        plt.title('Simulation Time vs Wall Clock Time')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()

                        break
            finally:
                if record_thread is not None:
                    stop_event.set()
                    capture_event.set()  # Wake the worker so it sees stop_event
                    record_thread.join()
                    print(f"Video saved: {self.record_output} ({frame_count} frames)")

    def render_cable(self, scene, positions, colors):
        """Draw the cable as capsule connectors and node spheres using transient scene geoms.

        Args:
            scene: MuJoCo mjvScene to append geoms to.
            positions: (N, 3) float64 array of node world positions from sync_display.
            colors: (N, 4) float32 RGBA array of per-node colors from sync_display.
        """
        n_visible = len(positions)
        if n_visible == 0:
            print("WARNING: no visible cable detected!")
            return

        spool_abs = 1 + self.anchor_idx + self.spool_idx  # +1 for attachment point at index 0
        identity_mat = np.eye(3, dtype=np.float64).flatten()
        sphere_size = np.array([CABLE_NODE_VISUAL_RADIUS, 0, 0], dtype=np.float64)

        # Free link from AV to spool
        if scene.ngeom < scene.maxgeom:
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, CABLE_VISUAL_RADIUS,
                                 self._get_av_pos(), positions[spool_abs])
            g.rgba[:] = COLOR_FREE_LINK
            g.label = ""
            scene.ngeom += 1

        # Draw chain segment connectors (absolute indices 0 to spool_abs-1)
        for abs_i in range(min(spool_abs, n_visible - 1)):
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, CABLE_VISUAL_RADIUS,
                                 positions[abs_i], positions[abs_i + 1])
            # Frozen segments (including attachment point) darker red, active segments orange
            if abs_i < 1 + self.anchor_idx:
                g.rgba[:] = COLOR_FROZEN_SEG
            else:
                g.rgba[:] = COLOR_ACTIVE_SEG
            g.label = ""
            scene.ngeom += 1

        # Draw node spheres
        for abs_i in range(n_visible):
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, sphere_size,
                                positions[abs_i], identity_mat, colors[abs_i])
            g.label = ""
            scene.ngeom += 1


if __name__ == "__main__":
    print("=" * 50)
    print("Gecko Lasso Simulation with MuJoCo")
    print("=" * 50)

    """
    ################
    ## PART 1: CAPTURING DIFFERENT SHAPES
    ################
    
    # Video #1: Detumbling cylindrical body
    sim = Simulation(sat_omega=1,
                     sat_rotation_axis=(0, 0.001, 1),
                     sat_attach_pos=(-1.8, 0, 0),
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=40,
                     record_output="BodyCaptureDemo_Cylinder.mp4",
                     sat_obj_path="Assets/Satellite_Cylinder.obj")
    sim.run()

    # Video #2: Detumbling sqaure cross section body
    sim = Simulation(sat_omega=1,
                     sat_rotation_axis=(0, 0.001, 1),
                     sat_attach_pos=(-1.8, 0, 0),
                     cable_node_radius=0.01,
                     cable_seg_len=0.25,
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=40,
                     record_output="BodyCaptureDemo_Square.mp4",
                     sat_obj_path="Assets/Satellite_Square.obj")
    sim.run()

    # Video #3: Detumbling I-beam cross section body
    sim = Simulation(sat_omega=1,
                     sat_rotation_axis=(0, 0.001, 1),
                     sat_attach_pos=(-1.8, 0, 0),
                     cable_node_radius=0.01,
                     cable_seg_len=0.25,
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=40,
                     record_output="BodyCaptureDemo_IBeam.mp4",
                     sat_obj_path="Assets/Satellite_IBeam.obj")
    sim.run()

    # Video #4: Detumbling snow man cross section body
    sim = Simulation(sat_omega=1,
                     sat_rotation_axis=(0, 0.001, 1),
                     sat_attach_pos=(-1.250, 0, 0),
                     cable_node_radius=0.01,
                     cable_seg_len=0.25,
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=40,
                     record_output="BodyCaptureDemo_Snowman.mp4",
                     sat_obj_path="Assets/Satellite_Snowman.obj")
    sim.run()

    # Video #5: Detumbling top hat body
    sim = Simulation(sat_omega=1,
                     sat_rotation_axis=(0, 0.001, 1),
                     sat_attach_pos=(-2.092, 0, 0),
                     cable_node_radius=0.01,
                     cable_seg_len=0.25,
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=40,
                     record_output="BodyCaptureDemo_Hat.mp4",
                     sat_obj_path="Assets/Satellite_Hat.obj")
    sim.run()

    # Video #6: Detumbling arm body
    sim = Simulation(sat_omega=1,
                     sat_rotation_axis=(0, 0.001, 1),
                     sat_attach_pos=(-1.5, 0, 0),
                     cable_node_radius=0.01,
                     cable_seg_len=0.25,
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=40,
                     record_output="BodyCaptureDemo_Arms.mp4",
                     sat_obj_path="Assets/Satellite_Arms.obj")
    sim.run()
    """

    ################
    ## PART 2: CAPTURING DIFFERENT SHAPES
    ################

    """
    # Video #6: Capturing and detumbling a spinning cylinder about main axis
    # Same as video #1
    sim = Simulation(sat_omega=1,
                     sat_rotation_axis=(0, 0.001, 1),
                     sat_attach_pos=(-1.8, 0, 0),
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=40,
                     record_output="BodyCaptureDemo_Cylinder.mp4",
                     sat_obj_path="Assets/Satellite_Cylinder.obj")
    sim.run()
    """

    """
    # Video #7: Detumbling cylindrical body
    sim = Simulation(sat_omega=1.5,
                     sat_rotation_axis=(0, 0, 1),
                     sat_attach_pos=(-1.8, 0, 0),
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=60,
                     record_output="DetumblingDemo_2DCylinder.mp4",
                     sat_obj_path="Assets/Satellite_Cylinder.obj")
    sim.run()

    # Video #8: Detumbling cylindrical body – minor procession
    sim = Simulation(sat_omega=1.5,
                     sat_rotation_axis=(0.03, 0, 1),
                     sat_attach_pos=(-1.8, 0, 0),
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=60,
                     record_output="DetumblingDemo_MinorProcession.mp4",
                     sat_obj_path="Assets/Satellite_Cylinder.obj")
    sim.run()

    # Video #8: Detumbling cylindrical body – minor procession
    sim = Simulation(sat_omega=1.5,
                     sat_rotation_axis=(0.09, 0, 1),
                     sat_attach_pos=(-1.8, 0, 0),
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=60,
                     record_output="DetumblingDemo_MajorProcession.mp4",
                     sat_obj_path="Assets/Satellite_Cylinder.obj")
    sim.run()

    # Video #9: Detumbling cylindrical body – major procession
    sim = Simulation(sat_omega=3,
                     sat_rotation_axis=(0, 0.5, 0.5),
                     sat_attach_pos=(-1.8, 0, 0),
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=60,
                     record_output="DetumblingDemo_OffAxis.mp4",
                     sat_obj_path="Assets/Satellite_Cylinder.obj")
    sim.run()

    # Video #10: Detumbling cylindrical body – major procession
    sim = Simulation(sat_omega=1.5,
                     sat_rotation_axis=(0.03, 0.5, 1),
                     sat_attach_pos=(-1.8, 0, 0),
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=60,
                     record_output="DetumblingDemo_OffsetCOM.mp4",
                     sat_obj_path="Assets/Satellite_CylinderOffsetCOMAxial.obj")
    sim.run()

    ################
    ## PART 3: INITIAL ATTACHMENT AGNOSTIC
    ################
    # Video 11: Initial anchor at the center of the satellite
    sim = Simulation(sat_omega=2,
                     sat_rotation_axis=(0, 0, 1),
                     sat_attach_pos=(-1.8, 0, 0),
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=70,
                     record_output="InitialAttachmentDemo_Center.mp4",
                     sat_obj_path="Assets/Satellite_Cylinder.obj")
    sim.run()

    # Video 12: Initial anchor at the top of the satellite
    sim = Simulation(sat_omega=2,
                     sat_rotation_axis=(0, 0, 1),
                     sat_attach_pos=(-1.8, 0, 4),
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=70,
                     record_output="InitialAttachmentDemo_Top.mp4",
                     sat_obj_path="Assets/Satellite_Cylinder.obj")
    sim.run()

    # Video 13: Initial anchor at the bottom of the satellite
    sim = Simulation(sat_omega=2,
                     sat_rotation_axis=(0, 0, 1),
                     sat_attach_pos=(-1.8, 0, -4),
                     av_init_pos=(0, 15, 0),
                     end_simulation_time=70,
                     record_output="InitialAttachmentDemo_Bottom.mp4",
                     sat_obj_path="Assets/Satellite_Cylinder.obj")
    sim.run()
    """
