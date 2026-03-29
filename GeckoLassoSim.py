"""
Gecko Lasso Simulation — simulation of a cable wrapping, capturing, and detumbling a satellite
M. Coughlin 2026 created for the BDML research group at Stanford University

This code is designed to use the MuJoCo native physics engine as much as possible

Requirements:
    pip install mujoco numpy matplotlib trimesh

Run:
    python GeckoLassoSim.py
"""

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
CABLE_NODE_RADIUS = 0.15  # Radius of the point masses on the cable on the display. Not physical! Units of m.

# Node and cable colors (RGBA)
COLOR_FROZEN = [1, 0, 0, 1]           # Red — frozen/welded nodes
COLOR_SPOOL = [0, 1, 0, 1]            # Green — spool node
COLOR_DAMPED = [0, 0.8, 1, 1]         # Cyan — damped nodes
COLOR_UNDAMPED = [1, 1, 0, 1]         # Yellow — undamped nodes
COLOR_FREE_LINK = [1.0, 0.4, 0.15, 1.0]    # Orange — free link from AV to spool
COLOR_FROZEN_SEG = [0.8, 0.1, 0.1, 1.0]    # Dark red — frozen chain segments
COLOR_ACTIVE_SEG = [1.0, 0.45, 0.15, 1.0]  # Orange — active chain segments
COLOR_INACTIVE = [0.5, 0.5, 0.5, 0]        # Invisible gray — inactive/pre-spawned nodes
COLOR_SATELLITE = [0.78, 0.82, 0.88, 1]    # Light gray-blue — satellite material
COLOR_AV = [0.3, 0.55, 0.95, 1]            # Blue — AV material
COLOR_GROUND = [0.2, 0.2, 0.2, 1]          # Dark gray — ground plane material

# Density of the satellite material. Right now, the density is homogenous. Non-homogenous densities are possible but
# require manually calculating the inertial tensor and center of mass and specifying them in the XML
SAT_MATERIAL_DENSITY = 7000  # Density of the satellite (kg/m^3). 7000 corresponds to aluminum.

# Contact solver parameters for cable-satellite collisions
CONTACT_MARGIN = 0.001        # Distance (m) at which contact constraints activate before geometric penetration
CONTACT_SOLREF_DAMPRATIO = 1  # Damping ratio for contact spring for contact forces (1 = critically damped)
CONTACT_SOLREF_STIFFNESS = 0.4  # Dimensionless constant relating contact timeconst to sqrt(mass/tension)

# (dmin, dmax, width, midpoint, power) — constant impedance
# dmin == dmax so the behavior scales properly regardless of cable tension (ie with proper values, the simulation will
# remain stable and well-behaves over tensions spanning at least 2 orders of magnitude
CONTACT_SOLIMP = (0.95, 0.95, 0.001, 0.5, 2)

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
RECOMPILE_BUFFER = 5

# The number of nodes, as measured from the last node touching the satellite, to leave undamped in the lateral
# direction. Note that axial damping is always applied to all tendons.
UNDAMPED_LATERAL_NODES_BEFORE_SAT = 4

# The desired ratio between the length of the free link and the total distance between the AV and the current departure
# point on the satellite
FREE_LINK_AV_SAT_DIST_RATIO = 0.7

# A new node spawns when the free link length is equal to the target free link length plus the equilibrium spacing
# between nodes multiplied by SPAWN_NEW_NODE_GROWTH_THRESHOLD. This allows for decoupling node spawn and despawn
# position, preventing nodes from repeatedly spawning and despawning due to oscillations around the spawn/despawn
# position. This should be >= 1 to be meaningful — otherwise, the spool will move further back than desired on the
# free link!
SPAWN_NEW_NODE_GROWTH_THRESHOLD = 1.1

# A similar parameter to SPAWN_NEW_NODE_GROWTH_THRESHOLD, except this time controlling when a node despawns. To be
# meaningful, it must be <= 1
DESPAWN_NODE_SHRINK_THRESHOLD = 0.1


def _rgba_str(color):
    """Convert an RGBA color list to a MuJoCo XML rgba string."""
    return f"{color[0]} {color[1]} {color[2]} {color[3]}"


def load_mesh(path: Path) -> Path:
    """Load a .obj mesh, make it watertight, center it on its center of mass, and export
    a cleaned copy for MuJoCo. This is necessary since .obj files exported from CAD
    software are not guaranteed to be "well-behaved".

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
    def __init__(self, sat_omega: float = 2, sat_rotation_axis: tuple = (0, 0.02, 1),
                 sat_attach_pos: tuple = (-1.8, 0, 0), av_init_pos: tuple = (-7.8, 20, -4),
                 thruster_react_func = lambda torque_vec: -1 * torque_vec, cable_tension: float = 10,
                 cable_stiffness: float = 7000, axial_damping_ratio: float = 5, lateral_damping_ratio: float = 0.7,
                 cable_pt_mass: float = 0.000259, cable_node_diameter: float = 0.01, cable_friction: tuple = (0.7, 0.2, 0.2),
                 cable_seg_len: float = 0.5, free_link_cable_ratio: float = 0.7, max_seg_num: int = 10000,
                 time_step: float = 0.00005, imp_ratio: float = 10, freeze_speed: float = 0.005,
                 freeze_time: float = 0.1, freeze_sample_interval: float = 0.002):
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
            cable_node_diameter: Diameter of each discretized cable node for physical simulation (m).
            cable_friction: (sliding, torsional, rolling) friction coefficients between cable nodes and the satellite
                surface.
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
        """

        # Save parameters for the simulation
        self.sat_omega = sat_omega
        self.sat_rotation_axis = np.array(sat_rotation_axis)
        self.sat_attach_pos = np.array(sat_attach_pos)
        self.av_init_pos = np.array(av_init_pos)
        self.thruster_react_func = thruster_react_func
        self.cable_tension = cable_tension
        self.cable_stiffness = cable_stiffness

        critical_damping = 2.0 * np.sqrt(2.0 * cable_tension * cable_pt_mass / cable_seg_len)
        self.cable_axial_damping = axial_damping_ratio * critical_damping
        self.cable_lateral_damping = lateral_damping_ratio * critical_damping

        self.cable_mass = cable_pt_mass
        self.cable_node_diameter = cable_node_diameter
        self.cable_friction = cable_friction
        self.cable_seg_len = cable_seg_len
        self.max_seg_num = max_seg_num
        self.time_step = time_step
        self.imp_ratio = imp_ratio

        self._freeze_threshold = freeze_speed
        # The number of measurements which must be below freeze_speed to freeze the node
        self._freeze_window_samples = int(freeze_time / freeze_sample_interval) + 1
        self._freeze_check_interval = freeze_sample_interval


        # Calculate the equilibrium length of the string
        equilibrium_cable_stretch = self.cable_tension / self.cable_stiffness
        self.seg_equilibrium_len = self.cable_seg_len + equilibrium_cable_stretch

        # Contact spring timeconst scales as sqrt(mass/tension) to maintain consistent penetration depth across
        # different tension and mass configurations
        contact_timeconst = CONTACT_SOLREF_STIFFNESS * np.sqrt(cable_pt_mass / cable_tension)
        self.contact_solref = (contact_timeconst, CONTACT_SOLREF_DAMPRATIO)

        # Cache mesh files to reduce computational cost of a model recompilation
        cwd = Path.cwd()
        self._sat_obj_file = load_mesh(cwd / "Assets" / "SatelliteNGPayload.obj")
        self._av_obj_file = load_mesh(cwd / "Assets" / "AV_1_0p5_0p5.obj")
        self._background_im_file = str(cwd / "Assets" / "Earth5.png")

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
        for i in range(self.num_init_segments - 1, self.compiled_seg_count - 1):
            tendon_id = self.cable_tendon_ids[i]
            self.model.tendon_stiffness[tendon_id] = 0
            self.model.tendon_damping[tendon_id] = 0

        # Store the state of the cable unspooling
        self.spool_idx = self.num_init_segments - 1
        self.active_count = self.num_init_segments

        self.anchor_idx = 0  # Absolute index of the first node in the physics model
        self._frozen_body_frame_pos = []  # List of satellite-body-frame positions of frozen nodes on the satellite
        self._freeze_vel_history = {}  # Dict of physics idx: deque of recent relative velocities

        self._next_freeze_check = 0
        self._pending_freeze_count = 0
        # Physics index of the node where damping begins. This is after the handful of nodes coming off the satellite
        # which are undamped
        self._damping_start = 0

        # Tension tracking
        # This is mainly for debugging, since issues tend to show up clearly in the cable tension
        self.tension_history = []
        self.time_history = []
        self._tension_record_interval = 1.0 / 200.0
        self._next_tension_record_time = 0.0

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
        self.cable_tendon_ids = []
        self.cable_jnt_ids = []
        for i in range(self.compiled_seg_count):
            self._phys_cable_body_arr[i] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"cable_{i}")
            self._phys_cable_geom_arr[i] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"cable_geom_{i}")
            self.cable_jnt_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cable_free_{i}"))

            if i < self.compiled_seg_count - 1:
                self.cable_tendon_ids.append(
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, f"cable_tendon_{i}"))

        self._phys_cyl_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")

        # Pre-computed DOF addresses for vectorized force application
        self._cable_dof_adrs = np.array([self.model.jnt_dofadr[jid] for jid in self.cable_jnt_ids], dtype=np.intp)
        self._sat_geom_id = self.model.body_geomadr[self._phys_cyl_body]

        # Reverse lookup: geom ID -> cable index for _get_cable_leave_pos
        self._geom_id_to_cable_idx = {gid: i for i, gid in enumerate(self._phys_cable_geom_arr)}

        # Per-step cache for _get_cable_leave_pos
        self._leave_pos_cache = None
        self._leave_pos_cache_time = -1.0
        self._cable_leave_idx = 0

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
        for i in range(max(self.active_count - 1, 0), self.compiled_seg_count - 1):
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

        total_visible = self.anchor_idx + self.active_count
        positions = np.empty((total_visible, 3), dtype=np.float64)
        colors = np.empty((total_visible, 4), dtype=np.float32)

        # Frozen nodes: transform stored body-frame positions to world frame
        sat_pos = self.data.xpos[self._phys_cyl_body]
        sat_quat = self.data.xquat[self._phys_cyl_body]

        world_pos = np.zeros(3)
        for f_idx in range(self.anchor_idx):
            mujoco.mju_rotVecQuat(world_pos, self._frozen_body_frame_pos[f_idx], sat_quat)
            positions[f_idx] = world_pos + sat_pos
            colors[f_idx] = COLOR_FROZEN

        # Active nodes: read positions from physics qpos
        for i in range(self.active_count):
            p_adr = self.model.jnt_qposadr[self.cable_jnt_ids[i]]
            abs_idx = self.anchor_idx + i
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
        # The anchor tendon has springlength=0, so its equilibrium stretch under cable_tension is
        # cable_tension / cable_stiffness. Offset all nodes by this amount so the anchor tendon starts at its
        # equilibrium length rather.
        anchor_equilibrium_stretch = self.cable_tension / self.cable_stiffness
        offsets = np.arange(num_segments) * self.seg_equilibrium_len + anchor_equilibrium_stretch
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
                <geom name="cable_geom_{i}" type="sphere" size="{self.cable_node_diameter}" mass="{self.cable_mass}"
                      friction="{self.cable_friction[0]} {self.cable_friction[1]} {self.cable_friction[2]}"
                      rgba="{color}" condim="6" contype="1" conaffinity="2"/>
                <site name="cable_site_{i}" pos="0 0 0" size="0.005"/>
            </body>
            """

            if i < num_segments - 1:  # There are only num_segments - 1 tendons (signpost problem)
                cable_tendons += f"""
                <spatial name="cable_tendon_{i}" springlength="{self.cable_seg_len}"
                        stiffness="{self.cable_stiffness}" damping="{self.cable_axial_damping}" limited="false">
                    <site site="cable_site_{i}"/>
                    <site site="cable_site_{i + 1}"/>
                </spatial>"""

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
                    <site name="cable_origin" pos="0 0 0" size="0.06" rgba="{_rgba_str(COLOR_SPOOL)}"/>
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
                        size="0.05" rgba="{_rgba_str(COLOR_FROZEN)}"/>
                </body>

            {cable_bodies}

            </worldbody>
            {"" if num_segments == 0 else f'''
            <tendon>
                <spatial name="cable_anchor_tendon" springlength="0"
                        stiffness="{self.cable_stiffness}" damping="{self.cable_axial_damping}" limited="false">
                    <site site="attachment_point"/>
                    <site site="cable_site_0"/>
                </spatial>
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
        return self.data.xpos[self._phys_cable_body_arr[idx]].copy()

    def _get_cable_leave_pos(self):
        """Find where the cable departs the satellite surface.

        Uses vectorized numpy on the contact arrays to find the highest-index cable node still touching the satellite
        mesh. Caches per step so only the first call is computationally intensive.
        """
        # This might seem like a lengthy and convoluted way to solve for the departure point. However, finding the
        # departure can be EXTREMELY computationally expensive (on the order of solving the physics itself) unless
        # the operation is carefully vectorized.

        # Return cached result if already computed this step
        if self._leave_pos_cache_time == self.data.time:
            return self._leave_pos_cache

        if self.data.ncon == 0:
            result = self.data.site_xpos[self.attachment_site_id].copy()
            self._cable_leave_idx = 0
            self._leave_pos_cache = result
            self._leave_pos_cache_time = self.data.time
            return result

        # Vectorized: read all contact geom pairs at once
        geom1 = self.data.contact.geom1[:self.data.ncon]
        geom2 = self.data.contact.geom2[:self.data.ncon]

        # Find contacts where one geom is the satellite
        sat_in_g1 = geom1 == self._sat_geom_id
        sat_in_g2 = geom2 == self._sat_geom_id

        # The other geom in each satellite contact
        other_geom = np.where(sat_in_g1, geom2, np.where(sat_in_g2, geom1, -1))

        # Build a set of active cable geom IDs for fast lookup
        cable_geom_set = self._phys_cable_geom_arr[:self.active_count]

        # Check which "other" geoms are cable geoms
        is_cable = np.isin(other_geom, cable_geom_set)

        if not np.any(is_cable):
            result = self.data.site_xpos[self.attachment_site_id].copy()
            self._cable_leave_idx = 0
        else:
            # Map geom IDs back to cable indices using the lookup table
            cable_geoms_in_contact = other_geom[is_cable]
            max_idx = max(self._geom_id_to_cable_idx.get(int(cg), -1) for cg in cable_geoms_in_contact)

            self._cable_leave_idx = max_idx
            result = self._get_node_pos(max_idx)

        self._leave_pos_cache = result
        self._leave_pos_cache_time = self.data.time
        return result

    def step(self):
        """Advance the simulation by one timestep: apply forces, step physics, and handle cable segment
        spawning/despawning, node freezing, and tension recording."""

        # Apply tension force on spool - must be done before mj_step
        self._apply_tension_and_forces()

        # Step forward
        mujoco.mj_step(self.model, self.data)

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

        # Check each node to test if it is frozen
        for i in range(self.active_count):
            # Cable node velocity
            cable_qvel_adr = self.model.jnt_dofadr[self.cable_jnt_ids[i]]
            v_cable = self.data.qvel[cable_qvel_adr:cable_qvel_adr + 3]

            # Satellite velocity at cable node's position
            pos_cable = self.data.xpos[self._phys_cable_body_arr[i]]
            v_sat_at_cable = v_sat + np.cross(omega_sat, pos_cable - pos_sat)

            # Relative velocity in the plane orthogonal to the cable direction. We only care about movement
            # in an orthogonal plane because a real cable is continuous, so the location of the nodes along the cable is
            # arbitrary and any component aligned with the cable is just moving the node along the cable!
            v_rel = v_cable - v_sat_at_cable
            pos_next = self.data.xpos[self._phys_cable_body_arr[i + 1]]
            cable_dir = pos_next - pos_cable
            cable_len = np.linalg.norm(cable_dir)
            if cable_len < 1e-9:
                print("WARNING: short cable detected in node freezing calculations.")

            cable_dir /= cable_len
            v_rel = v_rel - np.dot(v_rel, cable_dir) * cable_dir
            speed_rel = np.linalg.norm(v_rel)

            # Append to rolling window
            if i not in self._freeze_vel_history:
                self._freeze_vel_history[i] = deque(maxlen=self._freeze_window_samples)
            self._freeze_vel_history[i].append(speed_rel)

            # Node is frozen only if the max over the full window is below threshold
            history = self._freeze_vel_history[i]
            if len(history) == self._freeze_window_samples and max(history) < self._freeze_threshold:
                self._pending_freeze_count += 1  # Update the number of nodes which are detected to be frozen

            else:
                break  # Must be contiguous from anchor

    def _record_tension(self):
        """Saves current tendon and free-link tension for plotting"""
        self.time_history.append(self.data.time)

        # Get tendon forces
        tensions = []
        for i in range(self.active_count - 1):
            tendon_id = self.cable_tendon_ids[i]
            length = self.data.ten_length[tendon_id]
            stretch = length - self.cable_seg_len
            tensions.append(self.cable_stiffness * stretch)

        # Add free link tension, which is always constant
        tensions.append(self.cable_tension)

        self.tension_history.append(tensions)

    def _apply_tension_and_forces(self):
        """Compute and apply all external forces for the current timestep: cable tension on the spool node,
        AV thruster reaction force, and lateral damping on affected cable nodes."""
        av_pos = self._get_av_pos()
        torque_vec = np.zeros(3, dtype=np.float64)
        qfrc_applied = np.zeros(self.model.nv, dtype=np.float64)

        # Apply full tension at the spool node only
        spool_pos = self.data.xpos[self._phys_cable_body_arr[self.spool_idx]]
        toward_av = av_pos - spool_pos
        dist = np.linalg.norm(toward_av)

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

        # Store body-frame positions of nodes being frozen
        for i in range(freeze_count):
            world_pos = self._get_node_pos(i)
            body_pos = self._world_to_sat_body_frame(world_pos)
            self._frozen_body_frame_pos.append(body_pos)

        # Place the new anchor so the anchor tendon (springlength=0) starts pre-loaded
        # with the equilibrium tension.  Offset from node 0's position toward the
        # satellite (along the local cable direction) by T/k so that the initial
        # tendon length equals cable_tension/cable_stiffness and the spring force
        # matches the cable_tension
        new_cable0_world_pos = self._get_node_pos(freeze_count)
        last_frozen_world_pos = self._get_node_pos(freeze_count - 1)
        cable_dir = new_cable0_world_pos - last_frozen_world_pos
        cable_dir_len = np.linalg.norm(cable_dir)
        if cable_dir_len < 1e-9:
            print("WARNING: short cable detected when freezing the anchor node!")

        cable_dir /= cable_dir_len
        preload_offset = self.cable_tension / self.cable_stiffness
        anchor_world_pos = new_cable0_world_pos - cable_dir * preload_offset
        self._current_anchor_body_pos = self._world_to_sat_body_frame(anchor_world_pos)

        # Update indices
        self.anchor_idx += freeze_count
        self.spool_idx -= freeze_count
        self.active_count -= freeze_count

        # Shift freeze tracking indices to match new physics model
        self._freeze_vel_history = {k - freeze_count: v
                                    for k, v in self._freeze_vel_history.items() if k >= freeze_count}
        self._pending_freeze_count = 0

        print(f"  [FREEZE] Froze {freeze_count} nodes, anchor_idx now {self.anchor_idx}, active: {self.active_count}")

    def _maybe_spawn_despawn(self):
        """Check if the free link between the spool and AV has grown long enough to warrant spawning a new cable
        segment or shrunk far enough that one cable segment must be removed. Trigger the correct action if so."""
        if self.active_count < self.max_seg_num:
            av_pos = self._get_av_pos()
            free_link_length = np.linalg.norm(av_pos - self._get_node_pos(self.spool_idx))

            # Calculate target free link length and see if another node is required
            cable_leave_pos = self._get_cable_leave_pos()
            target_free_link = np.linalg.norm(av_pos - cable_leave_pos) * FREE_LINK_AV_SAT_DIST_RATIO

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

        # Activate tendon between old spool and new segment
        tendon_id = self.cable_tendon_ids[self.spool_idx]
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

        # Deactivate the tendon between new_spool and old_spool
        tendon_id = self.cable_tendon_ids[new_spool]
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
        """Plot the tension history of individual cable links and the spool constraint over time. Mainly intended for
        debugging."""
        plt.figure(figsize=(14, 8), dpi=500)
        num_constraints = 50

        num_to_plot = num_constraints
        for j in range(num_to_plot):
            constraint_idx = num_constraints - num_to_plot + j
            data = []
            for tensions in self.tension_history:
                if constraint_idx < len(tensions) - 1:
                    data.append(tensions[constraint_idx])
                else:
                    data.append(0)
            plt.plot(self.time_history[:len(data)], data, label=f'Link {constraint_idx}', alpha=0.7)

        spool_data = [t[-1] for t in self.tension_history]
        plt.plot(self.time_history[:len(spool_data)], spool_data,
                 label='Spool constraint force', linewidth=2, color='red')

        plt.axhline(y=self.cable_tension, color='gray', linestyle='--', label='Applied tension')

        plt.xlabel('Time (s)')
        plt.ylabel('Tension (N)')
        plt.title('Link Tension Over Time')
        plt.ylim(bottom=0)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
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

        lines = [
            "--- Initial Parameters ---",
            f"Rot Axis: ({rot_norm[0]:.2f}, {rot_norm[1]:.2f}, {rot_norm[2]:.2f})",
            f"Sat. Angular Vel: {self.sat_omega:.2f} rad/s",
            f"Cable Tension: {self.cable_tension:.1f} N",
            f"AV Pos: ({self.av_init_pos[0]:.2f}, {self.av_init_pos[1]:.2f}, {self.av_init_pos[2]:.2f})",
            f"Attach Pt: ({self.sat_attach_pos[0]:.2f}, {self.sat_attach_pos[1]:.2f}, {self.sat_attach_pos[2]:.2f})",
            f"Sat Mass: {self.sat_mass:.0f} kg",
            f"AV Mass: {self.av_mass:.0f} kg",
            "",
            "--- Live Stats ---",
            f"Time: {self.data.time:.1f} s",
            f"Thruster Force: ({self.current_thruster_force[0]:.1f}, {self.current_thruster_force[1]:.1f},"
            + f"{self.current_thruster_force[2]:.1f})",
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
        base_pos = lookat + right * half_width * 0.45 + up * half_height * 0.75
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
                np.array([0.001, 0, 0], dtype=np.float64),
                pos,
                np.eye(3, dtype=np.float64).flatten(),
                np.array([0, 0, 0, 0], dtype=np.float32),
            )
            g.label = text
            scene.ngeom += 1

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

        spool_abs = self.anchor_idx + self.spool_idx
        identity_mat = np.eye(3, dtype=np.float64).flatten()
        sphere_size = np.array([CABLE_NODE_RADIUS, 0, 0], dtype=np.float64)

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
            # Frozen segments darker red, active segments orange
            if abs_i < self.anchor_idx:
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

    view_tendons = False
    sim = Simulation()

    with mujoco.viewer.launch_passive(sim.display_model, sim.display_data) as viewer:
        viewer.cam.azimuth = -0.8289683948863515
        viewer.cam.elevation = -21.25310724431816
        viewer.cam.distance = 34.68901593838663
        viewer.cam.lookat[:] = [0, 5, 0]

        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = view_tendons
        time.sleep(1)  # Pause to allow for looking at the scene before the simulation starts and making sure it looks good

        render_interval = 1.0 / 60.0  # Sync display at ~60 Hz real time
        last_render_wall = time.monotonic()
        wall_start = last_render_wall

        # Track sim time vs wall time for performance plotting
        perf_sim_times = []
        perf_wall_times = []
        perf_record_interval = 0.1  # Record every 0.1s sim time
        next_perf_record = 0.0

        while viewer.is_running():
            # Run physics steps until wall clock says it's time to render
            now = time.monotonic()
            while now - last_render_wall < render_interval:
                sim.step()
                now = time.monotonic()

                # Record performance data at fixed sim-time intervals
                # This is used to benchmark simulation speed and confirm the effectiveness of optimizations
                if sim.data.time >= next_perf_record:
                    perf_sim_times.append(sim.data.time)
                    perf_wall_times.append(now - wall_start)
                    next_perf_record = sim.data.time + perf_record_interval

            last_render_wall = now

            # Sync display once per render frame
            node_positions, node_colors = sim.sync_display()

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                sim.render_cable(viewer.user_scn, node_positions, node_colors)
                sim.render_overlay(viewer.user_scn, viewer)

            # Plot tension
            if sim.data.time > 5 and not hasattr(sim, 'plotted'):
                sim.plot_tension()
                sim.plotted = True

            # Plot performance
            if sim.data.time > 60 and not hasattr(sim, 'perf_plotted'):
                sim.perf_plotted = True
                plt.figure(figsize=(10, 6))
                plt.plot(perf_wall_times, perf_sim_times, linewidth=2)
                plt.xlabel('Wall Clock Time (s)')
                plt.ylabel('Simulation Time (s)')
                plt.title('Simulation Time vs Wall Clock Time')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

            # Upper limit on simulation time
            if sim.data.time > 480:
                exit()

            viewer.sync()
