# DAMPING


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
from time import sleep

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import trimesh


#########
# CONFIGURATION
# Global variables.... these should probably be moved to be parameters
#########

show_inactive_seg = False
ENABLE_RECOMPILE = True  # Set True to allow runtime XML recompilation (slower but reclaims DOFs)


def load_mesh(path: Path):
    mesh = trimesh.load(path, force="mesh")
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.apply_translation(-mesh.center_mass)
    temp_export_file = path.parent / ("._Repositioned_" + path.name)
    mesh.export(temp_export_file)

    return temp_export_file


class Simulation:
    def __init__(self, sat_omega: float = 2, sat_rotation_axis: tuple = (0, 0.05, 1),
                 sat_attach_pos: tuple = (-1.8, 0, -3), av_init_pos: tuple = (-7.8, 20, 5),
                 thruster_react_func = lambda torque_vec: -1 * torque_vec, cable_tension: float = 10,
                 cable_stiffness: float = 7000, cable_damping: float = 50, cable_pt_mass: float = 0.000259,
                 cable_friction: tuple = (0.7, 0.2, 0.2), cable_seg_len: float = 0.5, max_seg_num: int = 1000,
                 time_step: float = 0.00005, imp_ratio: float = 5):  # time_step: float = 0.000005

        # Density of Nylon is ~1100 kg/m^3; assume 0.001 M diameter

        # Save parameters for the simulation
        self.sat_omega = sat_omega
        self.sat_rotation_axis = np.array(sat_rotation_axis)
        self.sat_attach_pos = np.array(sat_attach_pos)
        self.av_init_pos = np.array(av_init_pos)
        self.thruster_react_func = thruster_react_func
        self.cable_tension = cable_tension
        self.cable_stiffness = cable_stiffness
        self.cable_damping = cable_damping
        self.cable_mass = cable_pt_mass
        self.cable_friction = cable_friction
        self.cable_seg_len = cable_seg_len
        self.max_seg_num = max_seg_num
        self.time_step = time_step
        self.imp_ratio = imp_ratio

        # Define visual parameters — enlarged & brightened for Zoom visibility
        self.cable_visual_radius = 0.07  # Increased from 0.03
        self.cable_node_visual_radius = 0.15  # Must be larger than cable_visual_radius to be visible

        # Calculate the equilibrium length of the string
        equilibrium_cable_stretch = self.cable_tension / self.cable_stiffness  # 0.001m
        self.seg_equilibrium_len = self.cable_seg_len + equilibrium_cable_stretch  # 0.301m

        # Cache mesh paths so _create_model_xml doesn't reload them on every recompile
        cwd = Path.cwd()
        self._sat_obj_file = load_mesh(cwd / "3DModels" / "SatelliteNGPayload.obj")
        self._av_obj_file = load_mesh(cwd / "3DModels" / "AV_1_0p5_0p5.obj")
        self._background_im_file = str(cwd / "3DModels" / "Earth5.png")

        # Recompilation buffer — how many extra segments to compile beyond active count
        self.recompile_buffer = 5

        # Compute initial segment count and direction (needed before first compile)
        anchor_av_init_dir = self.av_init_pos - self.sat_attach_pos
        init_cable_max_len = np.linalg.norm(anchor_av_init_dir)
        anchor_av_init_dir /= init_cable_max_len
        # Chain covers (1 - free_link_fraction) of the initial cable length.
        # free_link_fraction=0.7 means 30% is chain, matching the spawn/despawn target.
        max_len_to_chain = init_cable_max_len * 0.3
        self.num_init_segments = max(4, int(max_len_to_chain / self.seg_equilibrium_len) + 1)

        # Compile only enough segments for the initial state plus buffer
        if ENABLE_RECOMPILE:
            self.compiled_seg_count = min(self.num_init_segments + self.recompile_buffer, self.max_seg_num)
        else:
            self.compiled_seg_count = self.max_seg_num

        # Current anchor position in satellite body frame — persists across recompiles
        self._current_anchor_body_pos = np.array(self.sat_attach_pos, dtype=np.float64)

        # Creates the lean physics model
        self.xml = self._create_model_xml(self.compiled_seg_count)
        self.model = mujoco.MjModel.from_xml_string(self.xml, None)
        self.data = mujoco.MjData(self.model)

        # Create full-size display model for the viewer (never recompiled)
        display_xml = self._create_model_xml(self.max_seg_num)
        self.display_model = mujoco.MjModel.from_xml_string(display_xml, None)
        self.display_data = mujoco.MjData(self.display_model)

        # Build ID caches for physics model
        self._rebuild_id_caches()

        # Build ID caches for display model
        self._display_cable_body_ids = []
        self._display_cable_geom_ids = []
        self._display_cable_jnt_ids = []
        self._display_cable_tendon_ids = []
        self._display_cylinder_jnt_id = mujoco.mj_name2id(self.display_model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_rotation")
        self._display_av_body_id = mujoco.mj_name2id(self.display_model, mujoco.mjtObj.mjOBJ_BODY, "av_body")
        self._display_attachment_body_id = mujoco.mj_name2id(self.display_model, mujoco.mjtObj.mjOBJ_BODY, "attachment_body")
        for i in range(self.max_seg_num):
            self._display_cable_body_ids.append(mujoco.mj_name2id(self.display_model, mujoco.mjtObj.mjOBJ_BODY, f"cable_{i}"))
            self._display_cable_geom_ids.append(mujoco.mj_name2id(self.display_model, mujoco.mjtObj.mjOBJ_GEOM, f"cable_geom_{i}"))
            self._display_cable_jnt_ids.append(mujoco.mj_name2id(self.display_model, mujoco.mjtObj.mjOBJ_JOINT, f"cable_free_{i}"))
            if i < self.max_seg_num - 1:
                self._display_cable_tendon_ids.append(
                    mujoco.mj_name2id(self.display_model, mujoco.mjtObj.mjOBJ_TENDON, f"cable_tendon_{i}"))

        # Deactivate all tendons in display model (we only use it for rendering positions)
        for i in range(self.max_seg_num - 1):
            tid = self._display_cable_tendon_ids[i]
            self.display_model.tendon_stiffness[tid] = 0
            self.display_model.tendon_damping[tid] = 0

        # Pre-compute numpy arrays for vectorized sync_display
        self._phys_cyl_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
        self._disp_cyl_body = mujoco.mj_name2id(self.display_model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
        self._disp_cable_body_arr = np.array(self._display_cable_body_ids, dtype=np.intp)
        self._disp_cable_geom_arr = np.array(self._display_cable_geom_ids, dtype=np.intp)

        # Give bodies the proper initial conditions
        qvel_addr = self.model.jnt_dofadr[self.cylinder_jnt_id]
        sat_rot_dir = self.sat_rotation_axis / np.linalg.norm(self.sat_rotation_axis)
        self.data.qvel[qvel_addr + 3:qvel_addr + 6] = self.sat_omega * sat_rot_dir

        # The anchor moves with the cylinder rotation
        init_vel_anchor = np.cross(self.sat_omega * sat_rot_dir, self.sat_attach_pos)

        # All cable nodes should have the same velocity component along the cable
        cable_direction = anchor_av_init_dir

        # Project anchor velocity onto cable direction - this is the "pay out" speed
        cable_axial_velocity = (init_vel_anchor @ cable_direction) * cable_direction

        # All nodes get this same axial velocity
        for string_node in range(self.num_init_segments):
            body_id = self.cable_jnt_ids[string_node]
            qvel_addr = self.model.jnt_dofadr[body_id]
            self.data.qvel[qvel_addr:qvel_addr + 3] = cable_axial_velocity

        # The anchor additionally has the perpendicular component (it's attached to rotating body)
        # So override node 0 with full anchor velocity
        qvel_addr = self.model.jnt_dofadr[self.cable_jnt_ids[0]]
        self.data.qvel[qvel_addr:qvel_addr + 3] = init_vel_anchor

        # Forward pass to initialize physics state consistently
        mujoco.mj_forward(self.model, self.data)

        # Compute body masses for display (sum of all geom masses in the body subtree)
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

        # Capstan freeze detection — nodes near the anchor that stop moving relative
        # to the satellite are frozen and dropped from the physics model on next recompile.
        self.anchor_idx = 0  # Absolute index of the first node in the physics model
        self._frozen_body_frame_pos = []  # List of body-frame positions on the satellite
        self._freeze_vel_history = {}  # Physics idx -> deque of recent relative velocities
        self._freeze_threshold = 0.005  # m/s — node considered frozen below this
        self._freeze_window_samples = 10  # Number of samples in the rolling window
        self._freeze_check_interval = 0.002  # Check every 50ms sim time (10 samples = 0.5s)
        self._next_freeze_check = 1.0  # Don't check in the first second
        self._pending_freeze_count = 0  # Contiguous frozen nodes from anchor end
        self._damping_start = 0  # First physics index receiving lateral damping

        # Tension tracking at 200 Hz simulation time
        self.tension_history = []
        self.time_history = []
        self._tension_record_interval = 1.0 / 200.0
        self._next_tension_record_time = 0.0

        # Initialize display model with current physics state
        self.sync_display()

    def body_inertia_world(self, body_name):
        # Get body ID
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        # 1. Local inertia (diagonal in body frame)
        # model.body_inertia gives (Ixx, Iyy, Izz)
        I_body = np.diag(self.model.body_inertia[body_id])

        # 2. Get body orientation in world frame
        # data.xmat is a flattened 3x3 rotation matrix
        R = self.data.xmat[body_id].reshape(3, 3)

        # 3. Rotate inertia tensor into world frame
        I_world = R @ I_body @ R.T

        return I_world


    def _rebuild_id_caches(self):
        self.cylinder_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_rotation")
        self.av_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "av_body")
        self.attachment_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "attachment_body")

        self.cable_body_ids = []
        self.cable_geom_ids = []
        self.cable_tendon_ids = []
        self.cable_jnt_ids = []
        for i in range(self.compiled_seg_count):
            self.cable_body_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"cable_{i}"))
            self.cable_geom_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"cable_geom_{i}"))
            self.cable_jnt_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cable_free_{i}"))

            if i < self.compiled_seg_count - 1:
                self.cable_tendon_ids.append(
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, f"cable_tendon_{i}"))

        # Numpy arrays for vectorized sync_display
        self._phys_cyl_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
        self._phys_cable_body_arr = np.array(self.cable_body_ids, dtype=np.intp)
        self._phys_cable_geom_arr = np.array(self.cable_geom_ids, dtype=np.intp)

        # Pre-computed DOF addresses for vectorized force application
        self._cable_dof_adrs = np.array([self.model.jnt_dofadr[jid] for jid in self.cable_jnt_ids], dtype=np.intp)

        # Satellite geom ID (first geom belonging to the cylinder body)
        cyl_body_id = self._phys_cyl_body
        self._sat_geom_id = self.model.body_geomadr[cyl_body_id]

        # Reverse lookup: geom ID -> cable index for _get_cable_leave_pos
        self._geom_id_to_cable_idx = {gid: i for i, gid in enumerate(self.cable_geom_ids)}

        # Per-step cache for _get_cable_leave_pos
        self._leave_pos_cache = None
        self._leave_pos_cache_time = -1.0
        self._cable_leave_idx = 0

    def _recompile_model(self, new_seg_count, anchor_body_frame_pos=None, frozen_this_recompile=0):
        old_model = self.model
        old_data = self.data
        old_compiled = self.compiled_seg_count

        # Save satellite state
        old_cyl_jnt_id = self.cylinder_jnt_id
        old_cyl_qpos_adr = old_model.jnt_qposadr[old_cyl_jnt_id]
        old_cyl_qvel_adr = old_model.jnt_dofadr[old_cyl_jnt_id]
        saved_cyl_qpos = old_data.qpos[old_cyl_qpos_adr:old_cyl_qpos_adr + 7].copy()
        saved_cyl_qvel = old_data.qvel[old_cyl_qvel_adr:old_cyl_qvel_adr + 6].copy()

        # Save active cable segment states, skipping freshly frozen nodes
        # Old physics indices frozen_this_recompile..frozen_this_recompile+active_count-1
        # map to new physics indices 0..active_count-1
        num_to_save = min(self.active_count, old_compiled - frozen_this_recompile)
        saved_cable_qpos = []
        saved_cable_qvel = []
        saved_cable_rgba = []
        for i in range(num_to_save):
            old_idx = i + frozen_this_recompile
            jnt_id = self.cable_jnt_ids[old_idx]
            qpos_adr = old_model.jnt_qposadr[jnt_id]
            qvel_adr = old_model.jnt_dofadr[jnt_id]
            saved_cable_qpos.append(old_data.qpos[qpos_adr:qpos_adr + 7].copy())
            saved_cable_qvel.append(old_data.qvel[qvel_adr:qvel_adr + 6].copy())
            saved_cable_rgba.append(old_model.geom_rgba[self.cable_geom_ids[old_idx]].copy())

        # Save active tendon stiffness/damping state (also offset by frozen count)
        saved_tendon_stiffness = []
        saved_tendon_damping = []
        for i in range(min(num_to_save - 1, len(self.cable_tendon_ids) - frozen_this_recompile)):
            old_tid_idx = i + frozen_this_recompile
            tid = self.cable_tendon_ids[old_tid_idx]
            saved_tendon_stiffness.append(old_model.tendon_stiffness[tid])
            saved_tendon_damping.append(old_model.tendon_damping[tid])

        saved_time = old_data.time

        # Recompile with new segment count
        self.compiled_seg_count = new_seg_count
        self.xml = self._create_model_xml(new_seg_count, anchor_body_frame_pos=anchor_body_frame_pos)
        self.model = mujoco.MjModel.from_xml_string(self.xml, None)
        self.data = mujoco.MjData(self.model)
        self._rebuild_id_caches()

        # Restore simulation time
        self.data.time = saved_time

        # Restore satellite state
        new_cyl_qpos_adr = self.model.jnt_qposadr[self.cylinder_jnt_id]
        new_cyl_qvel_adr = self.model.jnt_dofadr[self.cylinder_jnt_id]
        self.data.qpos[new_cyl_qpos_adr:new_cyl_qpos_adr + 7] = saved_cyl_qpos
        self.data.qvel[new_cyl_qvel_adr:new_cyl_qvel_adr + 6] = saved_cyl_qvel

        # Restore cable segment states
        for i in range(num_to_save):
            jnt_id = self.cable_jnt_ids[i]
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            qvel_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qpos[qpos_adr:qpos_adr + 7] = saved_cable_qpos[i]
            self.data.qvel[qvel_adr:qvel_adr + 6] = saved_cable_qvel[i]
            self.model.geom_rgba[self.cable_geom_ids[i]] = saved_cable_rgba[i]

        # Restore tendon stiffness/damping for active tendons
        for i in range(len(saved_tendon_stiffness)):
            tid = self.cable_tendon_ids[i]
            self.model.tendon_stiffness[tid] = saved_tendon_stiffness[i]
            self.model.tendon_damping[tid] = saved_tendon_damping[i]

        # Deactivate tendons for segments beyond active count
        for i in range(max(self.active_count - 1, 0), self.compiled_seg_count - 1):
            if i >= len(saved_tendon_stiffness):  # Don't overwrite already-restored tendons
                tid = self.cable_tendon_ids[i]
                self.model.tendon_stiffness[tid] = 0
                self.model.tendon_damping[tid] = 0

        # Recompute body masses for display
        sat_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
        self.sat_mass = self.model.body_subtreemass[sat_body_id]
        self.av_mass = self.model.body_subtreemass[self.av_body_id]

        mujoco.mj_forward(self.model, self.data)

        print(f"  [RECOMPILE] {old_compiled} -> {new_seg_count} segments (active: {self.active_count})")

    def sync_display(self):
        """Copy physics qpos into display model and run kinematics to update all transforms."""
        # Copy satellite qpos
        phys_cyl_qadr = self.model.jnt_qposadr[self.cylinder_jnt_id]
        disp_cyl_qadr = self.display_model.jnt_qposadr[self._display_cylinder_jnt_id]
        self.display_data.qpos[disp_cyl_qadr:disp_cyl_qadr + 7] = self.data.qpos[phys_cyl_qadr:phys_cyl_qadr + 7]

        # Place frozen nodes by transforming stored body-frame positions to world frame
        sat_pos = self.data.xpos[self._phys_cyl_body]
        sat_quat = self.data.xquat[self._phys_cyl_body]
        for f_idx in range(self.anchor_idx):
            body_frame_pos = self._frozen_body_frame_pos[f_idx]
            world_pos = np.zeros(3)
            mujoco.mju_rotVecQuat(world_pos, body_frame_pos, sat_quat)
            world_pos += sat_pos
            d_adr = self.display_model.jnt_qposadr[self._display_cable_jnt_ids[f_idx]]
            self.display_data.qpos[d_adr:d_adr + 3] = world_pos
            self.display_data.qpos[d_adr + 3:d_adr + 7] = sat_quat

        # Copy active cable segment qpos (physics idx i -> display idx anchor_idx + i)
        for i in range(self.active_count):
            p_adr = self.model.jnt_qposadr[self.cable_jnt_ids[i]]
            disp_abs_idx = self.anchor_idx + i
            d_adr = self.display_model.jnt_qposadr[self._display_cable_jnt_ids[disp_abs_idx]]
            self.display_data.qpos[d_adr:d_adr + 7] = self.data.qpos[p_adr:p_adr + 7]

        # Update all body/geom/site transforms from qpos (much lighter than mj_forward)
        mujoco.mj_kinematics(self.display_model, self.display_data)

        # Show frozen nodes in a distinct color (red = frozen/welded)
        for f_idx in range(self.anchor_idx):
            self.display_model.geom_rgba[self._display_cable_geom_ids[f_idx]] = [1, 0, 0, 1]

        # Color active segments by role: yellow (undamped), cyan (damped), green (spool)
        for i in range(self.active_count):
            disp_abs_idx = self.anchor_idx + i
            if i == self.spool_idx:
                color = [0, 1, 0, 1]  # Green spool
            elif i >= self._damping_start:
                color = [0, 0.8, 1, 1]  # Cyan damped
            else:
                color = [1, 1, 0, 1]  # Yellow undamped
            self.display_model.geom_rgba[self._display_cable_geom_ids[disp_abs_idx]] = color

        # Hide inactive display segments beyond active range
        first_inactive = self.anchor_idx + self.active_count
        if first_inactive < self.max_seg_num:
            inactive_gids = self._disp_cable_geom_arr[first_inactive:]
            self.display_model.geom_rgba[inactive_gids] = [0.5, 0.5, 0.5, 0]

    def _create_model_xml(self, num_segments, anchor_body_frame_pos=None):
        # Anchor position in satellite body frame (where cable_0 attaches)
        attach_pos = anchor_body_frame_pos if anchor_body_frame_pos is not None else self._current_anchor_body_pos

        # Vector from attachment toward AV
        anchor_av_init_dir = self.av_init_pos - self.sat_attach_pos
        init_cable_max_len = np.linalg.norm(anchor_av_init_dir)
        anchor_av_init_dir /= init_cable_max_len

        # Calculate positions for each node at equilibrium spacing
        active_positions = []
        for i in range(0, self.num_init_segments):
            pos = self.sat_attach_pos + anchor_av_init_dir * (i * self.seg_equilibrium_len)
            active_positions.append(pos)

        # Build cable bodies with sites
        cable_bodies = ""
        cable_tendons = ""
        for i in range(num_segments):
            if i < self.num_init_segments:
                pos = active_positions[i]
                color = "1 1 0 1"
            else:
                pos = self.av_init_pos + ((i - self.num_init_segments) * self.seg_equilibrium_len) * anchor_av_init_dir
                color = "0.5 0.5 0.5 0"

            cable_bodies += f"""
            <body name="cable_{i}" pos="{pos[0]} {pos[1]} {pos[2]}">
                <freejoint name="cable_free_{i}"/>
                <geom name="cable_geom_{i}" type="sphere" size="{self.cable_node_visual_radius}" mass="{self.cable_mass}"
                      friction="{self.cable_friction[0]} {self.cable_friction[1]} {self.cable_friction[2]}"
                      rgba="{color}" condim="6" contype="1" conaffinity="2"/>
                <site name="cable_site_{i}" pos="0 0 0" size="0.005"/>
            </body>
            """

            if i < num_segments - 1:  # There are only num_segments - 1 tendons (signpost problem)
                cable_tendons += f"""
                <spatial name="cable_tendon_{i}" springlength="{self.cable_seg_len}"
                        stiffness="{self.cable_stiffness}" damping="{self.cable_damping}" limited="false">
                    <site site="cable_site_{i}"/>
                    <site site="cable_site_{i + 1}"/>
                </spatial>"""

        sat_obj_file = self._sat_obj_file
        av_obj_file = self._av_obj_file
        background_im_file = self._background_im_file

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
                <!--texture type="skybox" builtin="gradient" rgb1="0.1 0.1 0.2" rgb2="0.3 0.3 0.4" width="512" height="512"/-->
                <texture type="skybox" file="{background_im_file}" rgb1="0.1 0.1 0.2" rgb2="0.3 0.3 0.4"/>
                
                <!--material name="gecko_lasso_image_material" texture="gecko_lasso_texture" specular="1" shininess="0.5"/-->
                
                <material name="cylinder_mat" rgba="0.3 0.5 0.8 0.8"/>
                <material name="sat_mat" rgba="0.78 0.82 0.88 1" emission="0.35" specular="0.3" shininess="0.15"/>
                <material name="av_mat" rgba="0.3 0.55 0.95 1" emission="0.3" specular="0.3" shininess="0.15"/>
                <material name="ground_mat" rgba="0.2 0.2 0.2 1" reflectance="0.1"/>
                <mesh name="sat_mesh" file="{sat_obj_file}" inertia="exact"/>
                <mesh name="av_mesh" file="{av_obj_file}" inertia="exact"/>
            </asset>

            <worldbody>
                <!--geom type="plane" size="20 20 0.1" pos="0 0 -10" material="ground_mat" contype="0" conaffinity="0"/-->

                 <body name="av_body" pos="{self.av_init_pos[0]} {self.av_init_pos[1]} {self.av_init_pos[2]}">
                    <!--freejoint name="av_free_joint"/-->
                    <geom type="mesh" mesh="av_mesh" contype="4" conaffinity="2" friction="0 0 0" condim="6" material="av_mat"/>
                    <site name="cable_origin" pos="0 0 0" size="0.06" rgba="0 1 0 1"/>
                </body>


                <body name="cylinder" pos="0 0 0">
                    <freejoint name="cylinder_rotation"/>
                    <geom type="mesh" mesh="sat_mesh" contype="2" conaffinity="1"
                        density="2700" friction="{self.cable_friction[0]} {self.cable_friction[1]} 
                        {self.cable_friction[2]}" condim="6" material="sat_mat" margin="0.002"
                        solimp="0.99 0.999 0.001 0.5 2" solref="0.0005 1"/>  # Margin sets collisions to start 1 mm away from surface

                    <site name="attachment_point" pos="{attach_pos[0]} {attach_pos[1]}
                            {attach_pos[2]}" size="0.05" rgba="1 0.5 0 1"/>
                    <body name="attachment_body" pos="{attach_pos[0]} {attach_pos[1]}
                            {attach_pos[2]}">
                        <geom name="attachment_geom" type="sphere" size="0.01" mass="0.001" contype="0" conaffinity="0" rgba="0 0 0 0"/>
                    </body>
                </body>

            {cable_bodies}

            </worldbody>

            <tendon>
                <spatial name="cable_anchor_tendon" springlength="0"
                        stiffness="{self.cable_stiffness}" damping="{self.cable_damping}" limited="false">
                    <site site="attachment_point"/>
                    <site site="cable_site_0"/>
                </spatial>
                {cable_tendons}
            </tendon>
        </mujoco>
        """

        return xml

    def _get_pos(self, idx):
        return self.data.xpos[self.cable_body_ids[idx]].copy()

    def _get_cable_leave_pos(self):
        """Find where the cable departs the satellite surface.

        Uses vectorized numpy on the contact arrays to find the highest-index
        cable node still touching the satellite mesh. Caches per step so the
        second call (from _maybe_despawn) is free.
        """
        # Return cached result if already computed this step
        t = self.data.time
        if self._leave_pos_cache_time == t:
            return self._leave_pos_cache

        ncon = self.data.ncon
        if ncon == 0:
            result = self.data.xpos[self.attachment_body_id].copy()
            self._cable_leave_idx = 0
            self._leave_pos_cache = result
            self._leave_pos_cache_time = t
            return result

        # Vectorized: read all contact geom pairs at once
        geom1 = self.data.contact.geom1[:ncon]
        geom2 = self.data.contact.geom2[:ncon]
        sat_id = self._sat_geom_id

        # Find contacts where one geom is the satellite
        sat_in_g1 = geom1 == sat_id
        sat_in_g2 = geom2 == sat_id

        # The other geom in each satellite contact
        other_geom = np.where(sat_in_g1, geom2, np.where(sat_in_g2, geom1, -1))

        # Build a set of active cable geom IDs for fast lookup
        cable_geom_set = self._phys_cable_geom_arr[:self.active_count]

        # Check which "other" geoms are cable geoms
        is_cable = np.isin(other_geom, cable_geom_set)

        if not np.any(is_cable):
            result = self.data.xpos[self.attachment_body_id].copy()
            self._cable_leave_idx = 0
        else:
            # Map geom IDs back to cable indices using the lookup table
            cable_geoms_in_contact = other_geom[is_cable]
            # Find the max cable index among contacts
            max_idx = 0
            for cg in cable_geoms_in_contact:
                idx = self._geom_id_to_cable_idx.get(int(cg), -1)
                if idx > max_idx:
                    max_idx = idx

            # Enforce monotonic increase — the leave point can only advance along the
            # cable, never jump backward (settled nodes may lose contacts transiently)
            max_idx = max(max_idx, self._cable_leave_idx)

            self._cable_leave_idx = max_idx
            result = self._get_pos(max_idx)

        self._leave_pos_cache = result
        self._leave_pos_cache_time = t
        return result

    def step(self):
        # Compute cable leave point once per step — used by tension, spawn, and despawn
        self._get_cable_leave_pos()

        # Apply tension force on spool - must be done before mj_step
        self._apply_tension_and_forces()

        # Step forward
        mujoco.mj_step(self.model, self.data)

        self._maybe_spawn()
        self._maybe_despawn()

        if self.data.time >= self._next_freeze_check:
            self._check_frozen_nodes()
            self._next_freeze_check = self.data.time + self._freeze_check_interval

        if self.data.time >= self._next_tension_record_time:
            self._record_tension()
            self._next_tension_record_time = self.data.time + self._tension_record_interval

    def _check_frozen_nodes(self):
        """Detect cable nodes near the anchor that have stopped moving relative to the satellite."""
        from collections import deque

        if self.active_count <= 4:
            return

        # Get satellite velocity state
        cyl_qvel_adr = self.model.jnt_dofadr[self.cylinder_jnt_id]
        v_sat = self.data.qvel[cyl_qvel_adr:cyl_qvel_adr + 3]
        omega_sat_local = self.data.qvel[cyl_qvel_adr + 3:cyl_qvel_adr + 6]
        R_sat = self.data.xmat[self._phys_cyl_body].reshape(3, 3)
        omega_sat = R_sat @ omega_sat_local
        pos_sat = self.data.xpos[self._phys_cyl_body]

        # Check nodes from anchor end (physics index 0) outward, up to half the chain
        max_check = min(self.active_count - 3, self.active_count // 2)
        contiguous_frozen = 0

        for i in range(max_check):
            # Cable node velocity
            cable_qvel_adr = self.model.jnt_dofadr[self.cable_jnt_ids[i]]
            v_cable = self.data.qvel[cable_qvel_adr:cable_qvel_adr + 3]

            # Satellite velocity at cable node's position
            pos_cable = self.data.xpos[self.cable_body_ids[i]]
            v_sat_at_cable = v_sat + np.cross(omega_sat, pos_cable - pos_sat)

            # Relative velocity in the plane orthogonal to the cable direction
            v_rel = v_cable - v_sat_at_cable
            pos_next = self.data.xpos[self.cable_body_ids[i + 1]]
            cable_dir = pos_next - pos_cable
            cable_len = np.linalg.norm(cable_dir)
            if cable_len > 1e-9:
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
                contiguous_frozen += 1
            else:
                break  # Must be contiguous from anchor

        self._pending_freeze_count = contiguous_frozen

    def _record_tension(self):
        self.time_history.append(self.data.time)

        # Get tendon forces
        tensions = []
        for i in range(self.active_count - 1):
            if i < len(self.cable_tendon_ids):
                tendon_id = self.cable_tendon_ids[i]
                length = self.data.ten_length[tendon_id]
                stretch = length - self.cable_seg_len
                tensions.append(self.cable_stiffness * stretch)
            else:
                tensions.append(0)

        # Add free link tension
        tensions.append(self.cable_tension)

        self.tension_history.append(tensions)

    def _apply_tension_and_forces(self):
        av_pos = np.array(self._get_av_pos(), dtype=np.float64)
        torque_vec = np.zeros(3, dtype=np.float64)
        qfrc_applied = np.zeros(self.model.nv, dtype=np.float64)

        # Apply full tension at the spool node only
        spool_pos = self.data.xpos[self.cable_body_ids[self.spool_idx]]
        toward_av = av_pos - spool_pos
        dist = np.linalg.norm(toward_av)
        if dist > 1e-9:
            direction = toward_av / dist
            force = direction * self.cable_tension
            body_id = self.cable_body_ids[self.spool_idx]
            mujoco.mj_applyFT(self.model, self.data, force, torque_vec, spool_pos, body_id, qfrc_applied)

        # Reaction force on the AV
        overall_dir = toward_av / dist if dist > 1e-9 else np.zeros(3)
        thruster_force = self.thruster_react_func(overall_dir * self.cable_tension)
        self.current_thruster_force = thruster_force.copy()
        mujoco.mj_applyFT(self.model, self.data, thruster_force, torque_vec,
                          av_pos, self.av_body_id, qfrc_applied)

        # Vectorized lateral damping: applied to nodes from leave_idx+4 through spool_idx.
        # Damps velocity orthogonal to the local tendon direction.
        lateral_damping = 0.1
        damping_start = min(self._cable_leave_idx + 4, self.spool_idx)
        n_damped = self.spool_idx - damping_start + 1

        if n_damped > 0 and self.spool_idx > 0:
            # Read all active positions in one shot
            body_ids = self._phys_cable_body_arr[:self.spool_idx + 1]
            all_pos = self.data.xpos[body_ids]  # (spool_idx+1, 3)

            # Slice to damped range
            d_start = damping_start
            d_end = self.spool_idx + 1  # exclusive

            # Compute tendon directions from neighbors for damped nodes
            # For interior nodes: dir = pos[i+1] - pos[i-1]
            # For first damped (if d_start==0): dir = pos[1] - pos[0]
            # For last damped (spool_idx): dir = pos[i] - pos[i-1]
            if d_start > 0:
                prev_pos = all_pos[d_start - 1:d_end - 1]  # i-1 for each damped node
            else:
                # First node uses forward difference
                prev_pos = np.empty((n_damped, 3))
                prev_pos[0] = all_pos[0]
                prev_pos[1:] = all_pos[d_start:d_end - 1]

            if d_end <= self.spool_idx:
                next_pos = all_pos[d_start + 1:d_end + 1]  # i+1 for each damped node
            else:
                # Last node uses backward difference
                next_pos = np.empty((n_damped, 3))
                next_pos[:-1] = all_pos[d_start + 1:d_end]
                next_pos[-1] = all_pos[self.spool_idx]

            tendon_dirs = next_pos - prev_pos  # (n_damped, 3)
            tendon_lens = np.linalg.norm(tendon_dirs, axis=1, keepdims=True)
            tendon_lens = np.maximum(tendon_lens, 1e-9)
            tendon_dirs /= tendon_lens

            # Read velocities for damped nodes via fancy indexing (no Python loop)
            dof_adrs = self._cable_dof_adrs[d_start:d_end]
            vel_indices = dof_adrs[:, None] + np.arange(3)  # (n_damped, 3) index array
            vels = self.data.qvel[vel_indices]  # (n_damped, 3)

            # Compute lateral velocity: v - (v·d)*d
            axial_proj = np.sum(vels * tendon_dirs, axis=1, keepdims=True)
            lateral_vels = vels - axial_proj * tendon_dirs

            # Damping forces
            damp_forces = -lateral_damping * lateral_vels  # (n_damped, 3)

            # Write directly to qfrc_applied (for freejoint bodies at worldbody, the
            # linear DOFs are at dof_adr:dof_adr+3 and mj_applyFT just writes there)
            for j in range(n_damped):
                a = int(dof_adrs[j])
                qfrc_applied[a:a + 3] += damp_forces[j]

            # Store damping range for color updates at render time
            self._damping_start = d_start

        self.data.qfrc_applied[:] = qfrc_applied

    def _get_av_pos(self):
        # start_addr = self.model.jnt_dofadr[self.av_joint_id]
        # return self.data.qpos[start_addr:start_addr+3]

        return self.av_init_pos[0], self.av_init_pos[1], self.av_init_pos[2]

    def _world_to_sat_body_frame(self, world_pos):
        """Convert a world-frame position to the satellite's body frame."""
        sat_pos = self.data.xpos[self._phys_cyl_body].copy()
        sat_quat = self.data.xquat[self._phys_cyl_body].copy()
        # Compute inverse rotation: negate the quaternion
        sat_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(sat_quat_inv, sat_quat)
        # Rotate (world_pos - sat_pos) by inverse quaternion
        delta = np.array(world_pos, dtype=np.float64) - sat_pos
        body_frame_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(body_frame_pos, delta, sat_quat_inv)
        return body_frame_pos

    def _freeze_anchor_nodes(self, freeze_count):
        """Freeze `freeze_count` nodes from the anchor end: store body-frame positions and
        prepare for recompilation that drops them from the physics model."""
        if freeze_count <= 0:
            return

        # Store body-frame positions of nodes being frozen
        for i in range(freeze_count):
            world_pos = self._get_pos(i)
            body_pos = self._world_to_sat_body_frame(world_pos)
            self._frozen_body_frame_pos.append(body_pos)

        # The new anchor is at the position of the first surviving node (old physics
        # index freeze_count), so that the anchor tendon doesn't yank it
        new_cable0_world_pos = self._get_pos(freeze_count)
        new_anchor_body_pos = self._world_to_sat_body_frame(new_cable0_world_pos)
        self._current_anchor_body_pos = new_anchor_body_pos

        # Update indices
        self.anchor_idx += freeze_count
        self.spool_idx -= freeze_count
        self.active_count -= freeze_count

        # Shift freeze tracking indices to match new physics model
        self._freeze_vel_history = {k - freeze_count: v
                                    for k, v in self._freeze_vel_history.items()
                                    if k >= freeze_count}
        self._pending_freeze_count = 0

        print(f"  [FREEZE] Froze {freeze_count} nodes, anchor_idx now {self.anchor_idx}, "
              f"active: {self.active_count}")

        return new_anchor_body_pos

    def _maybe_spawn(self):
        if self.active_count < self.max_seg_num:
            spool_pos = self._get_pos(self.spool_idx)
            av_pos = np.array(self._get_av_pos())

            free_link_length = np.linalg.norm(av_pos - spool_pos)

            # Target free link is 90% of the distance from the cable's departure
            # point on the satellite surface to the AV
            cable_leave_pos = self._get_cable_leave_pos()
            target_free_link = np.linalg.norm(av_pos - cable_leave_pos) * 0.7

            # Spawn only when free link has grown more than 1.1 segments above target
            if free_link_length > target_free_link + self.seg_equilibrium_len * 1.1:
                print(f"\n=== SPAWN TRIGGER t={self.data.time:.4f}s ===")
                print(f"Free link: {free_link_length:.4f}m > target {target_free_link:.4f}m + {self.seg_equilibrium_len * 1.1:.4f}m")
                self._spawn_segment()

    def _spawn_segment(self):
        # Forward the physics state. This must be done at the start AND at the end to get the proper location for the
        # old spool
        mujoco.mj_forward(self.model, self.data)

        new_idx = self.active_count

        # Recompile if we've run out of compiled segments — also drop frozen nodes
        if new_idx >= self.compiled_seg_count:
            anchor_pos = None
            frozen_now = 0
            if self._pending_freeze_count > 0:
                frozen_now = self._pending_freeze_count
                anchor_pos = self._freeze_anchor_nodes(frozen_now)
                # After freeze, active_count changed, recalculate
                new_idx = self.active_count

            if ENABLE_RECOMPILE:
                new_count = min(new_idx + self.recompile_buffer, self.max_seg_num)
                self._recompile_model(new_count, anchor_body_frame_pos=anchor_pos,
                                      frozen_this_recompile=frozen_now)
                mujoco.mj_forward(self.model, self.data)

        old_spool = self.spool_idx

        old_spool_pos = self._get_pos(old_spool)

        # Get old spool velocity from qvel
        old_joint_id = self.cable_jnt_ids[old_spool]
        old_qvel_adr = self.model.jnt_dofadr[old_joint_id]

        # Get new segment's joint addresses
        qpos_adr = self.model.jnt_qposadr[self.cable_jnt_ids[new_idx]]
        qvel_adr = self.model.jnt_dofadr[self.cable_jnt_ids[new_idx]]

        # Direction from old spool toward origin
        toward_origin = self._get_av_pos() - old_spool_pos
        toward_origin_dir = toward_origin / np.linalg.norm(toward_origin)

        # Position new spool at equilibrium distance from old spool, toward origin
        spawn_pos = old_spool_pos + toward_origin_dir * self.seg_equilibrium_len

        # Set new segment position and orientation
        self.data.qpos[qpos_adr:qpos_adr+3] = spawn_pos
        self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

        # To result in smoother spawning, we will give the cable an initial velocity
        self.data.qvel[qvel_adr:qvel_adr+3] = self.data.qvel[old_qvel_adr:old_qvel_adr+3]

        # Activate tendon between old spool and new segment
        tendon_id = self.cable_tendon_ids[old_spool]
        self.model.tendon_stiffness[tendon_id] = self.cable_stiffness
        self.model.tendon_damping[tendon_id] = self.cable_damping

        # Update visual colors (keep invisible — cable rendered by render_cable)
        self.model.geom_rgba[self.cable_geom_ids[new_idx]] = [0, 1, 0, 1]  # Green spool node
        self.model.geom_rgba[self.cable_geom_ids[old_spool]] = [1, 1, 0, 1]  # Yellow chain node

        # Update spool index and active count
        self.spool_idx = new_idx
        self.active_count = new_idx + 1

        # Now that the new spool is spawned, forward the state again. Forwarding twice is required to reduce tension
        # spikes
        mujoco.mj_forward(self.model, self.data)

    def _maybe_despawn(self):
        if self.spool_idx <= 3:  # Keep at least 4 segments
            return

        spool_pos = self._get_pos(self.spool_idx)
        av_pos = np.array(self._get_av_pos())

        free_link_length = np.linalg.norm(av_pos - spool_pos)

        # Target free link is 90% of distance from cable departure point to AV
        cable_leave_pos = self._get_cable_leave_pos()
        target_free_link = np.linalg.norm(av_pos - cable_leave_pos) * 0.7

        # Despawn when free link has shrunk more than 0.1 segments below target
        if free_link_length < target_free_link - self.seg_equilibrium_len * 0.1:
            print(f"\n=== DESPAWN TRIGGER t={self.data.time:.4f}s ===")
            print(f"Free link: {free_link_length:.4f}m < target {target_free_link:.4f}m - {self.seg_equilibrium_len * 0.1:.4f}m")
            self._despawn_segment()

    def _despawn_segment(self):
        # Forward the physics state first
        mujoco.mj_forward(self.model, self.data)

        old_spool = self.spool_idx
        new_spool = old_spool - 1  # Previous segment becomes the new spool

        print(f"DESPAWN: old_spool={old_spool}, new_spool={new_spool}")

        # Deactivate the tendon between new_spool and old_spool
        tendon_id = self.cable_tendon_ids[new_spool]
        self.model.tendon_stiffness[tendon_id] = 0
        self.model.tendon_damping[tendon_id] = 0

        # Disable contacts on the despawned body so it can't interfere
        old_geom_id = self.cable_geom_ids[old_spool]
        self.model.geom_contype[old_geom_id] = 0
        self.model.geom_conaffinity[old_geom_id] = 0

        # Move the old spool segment far away and zero its velocity
        old_spool_jnt_id = self.cable_jnt_ids[old_spool]
        qpos_adr = self.model.jnt_qposadr[old_spool_jnt_id]
        qvel_adr = self.model.jnt_dofadr[old_spool_jnt_id]

        # Place it far from anything (behind AV, one segment length per inactive body)
        av_pos = np.array(self._get_av_pos(), dtype=np.float64)
        inactive_offset = (old_spool - self.active_count + 1) * self.seg_equilibrium_len
        inactive_pos = av_pos + np.array([0, inactive_offset + 50, 0])

        self.data.qpos[qpos_adr:qpos_adr + 3] = inactive_pos
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0

        # Update visual colors
        self.model.geom_rgba[old_geom_id] = [0.5, 0.5, 0.5, 0]  # Hide despawned node
        self.model.geom_rgba[self.cable_geom_ids[new_spool]] = [0, 1, 0, 1]  # Green spool node

        # Update spool index and active count
        self.spool_idx = new_spool
        self.active_count = old_spool  # active_count is now old_spool (not old_spool + 1)

        print(f"  new spool_idx: {self.spool_idx}, active_count: {self.active_count}")

        # Recompile to shed excess DOFs if we have too many unused segments
        if ENABLE_RECOMPILE and self.compiled_seg_count - self.active_count > 2 * self.recompile_buffer:
            new_count = self.active_count + self.recompile_buffer
            self._recompile_model(new_count)

        # Allow the leave index to drop by 1 since the spool retracted
        if self._cable_leave_idx > 0:
            self._cable_leave_idx -= 1

        # Forward again to update state
        mujoco.mj_forward(self.model, self.data)


    def plot_tension(self):
        plt.figure(figsize=(14, 8))

        final_tensions = self.tension_history[-1]
        # num_constraints = len(final_tensions) - 1
        num_constraints = 50

        # num_to_plot = min(5, num_constraints)
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

        plt.axhline(y=self.cable_tension, color='gray', linestyle='--', label='Applied tension (10N)')

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

        Computes label positions dynamically from the current camera state so they
        always appear on the right side of the screen, even if the user orbits, zooms,
        or resizes the window.
        """
        # Normalized rotation axis for display
        rot_norm = self.sat_rotation_axis / np.linalg.norm(self.sat_rotation_axis)

        tf = self.current_thruster_force
        tf_mag = np.linalg.norm(tf)

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
            # f"Active Segs: {self.active_count} / {self.max_seg_num}",
            f"Thruster Force: ({tf[0]:.1f}, {tf[1]:.1f}, {tf[2]:.1f})",
            # f"Thruster Mag: {tf_mag:.3f} N",
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
            right = np.array([1.0, 0.0, 0.0])
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

    def _get_display_pos(self, abs_idx):
        """Get world position of absolute cable index from display model."""
        return self.display_data.xpos[self._display_cable_body_ids[abs_idx]].copy()

    def render_cable(self, scene):
        # Absolute index of spool in display model
        spool_abs = self.anchor_idx + self.spool_idx
        spool_pos = self._get_display_pos(spool_abs)

        # Free link from AV to spool
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, self.cable_visual_radius, self._get_av_pos(), spool_pos)
        g.rgba[:] = [1.0, 0.4, 0.15, 1.0]
        g.label = ""
        scene.ngeom += 1

        # Draw all chain segments: frozen + active (absolute indices 0 to spool_abs-1)
        for abs_i in range(spool_abs):
            p1 = self._get_display_pos(abs_i)
            p2 = self._get_display_pos(abs_i + 1)

            g = scene.geoms[scene.ngeom]
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, self.cable_visual_radius, p1, p2)
            # Frozen segments in darker red, active in orange
            if abs_i < self.anchor_idx:
                g.rgba[:] = [0.8, 0.1, 0.1, 1.0]
            else:
                g.rgba[:] = [1.0, 0.45, 0.15, 1.0]
            g.label = ""
            scene.ngeom += 1


if __name__ == "__main__":
    print("=" * 50)
    print("Gecko Lasso Simulation with MuJoCo")
    print("=" * 50)

    # sim = Simulation()
    use_interactive_viewer = True
    view_tendons = False

    sim = Simulation()
    # all_simulations = [,
    #                    Simulation(sat_omega=1, sat_rotation_axis=(0, 0, 1)),
    #                    Simulation(sat_omega=1, sat_rotation_axis=(0.1, 0, 1))]
    # sim_num = 0

    if use_interactive_viewer:
        with mujoco.viewer.launch_passive(sim.display_model, sim.display_data) as viewer:
            viewer.cam.azimuth = -0.8289683948863515
            viewer.cam.elevation = -21.25310724431816
            viewer.cam.distance = 34.68901593838663
            viewer.cam.lookat[:] = [0, 5, 0]

            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = view_tendons
            sleep(1)

            import time as _time
            render_interval = 1.0 / 60.0  # Sync display at ~60 Hz real time
            last_render_wall = _time.monotonic()
            wall_start = last_render_wall

            # Track sim time vs wall time for performance plotting
            perf_sim_times = []
            perf_wall_times = []
            perf_record_interval = 0.1  # Record every 0.1s sim time
            next_perf_record = 0.0

            while viewer.is_running():
                # Run physics steps until wall clock says it's time to render
                now = _time.monotonic()
                while now - last_render_wall < render_interval:
                    sim.step()
                    now = _time.monotonic()

                    # Record performance data at fixed sim-time intervals
                    if sim.data.time >= next_perf_record:
                        perf_sim_times.append(sim.data.time)
                        perf_wall_times.append(now - wall_start)
                        next_perf_record = sim.data.time + perf_record_interval

                last_render_wall = now

                # Sync display once per render frame
                sim.sync_display()

                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    sim.render_cable(viewer.user_scn)
                    sim.render_overlay(viewer.user_scn, viewer)

                if sim.data.time > 5 and not hasattr(sim, 'plotted'):
                    sim.plot_tension()
                    sim.plotted = True

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

                if sim.data.time > 480:
                    exit()

                viewer.sync()

    """
    else:
        frame_x = 3840
        frame_y = 2160
        fps = 60
        frame_interval = 1.0 / fps
        last_frame_time = 0.0
        video_index = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = f"cable_simulation_{video_index}.mp4"
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_x, frame_y))
        frame_count = 0

        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            # Set camera
            viewer.cam.azimuth = -10.5
            viewer.cam.elevation = -5
            viewer.cam.distance = 8.7
            viewer.cam.lookat[:] = [-20.9, 5.5, 2.3]

            while viewer.is_running() and sim.data.time < 120:
                sim.step()

                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    sim.render_cable(viewer.user_scn)

                # Capture frame at 60 FPS
                if sim.data.time - last_frame_time >= frame_interval:
                    # Read pixels from viewer

                    # Resize if needed
                    if pixels.shape[1] != frame_x or pixels.shape[0] != frame_y:
                        pixels = cv2.resize(pixels, (frame_x, frame_y), interpolation=cv2.INTER_LANCZOS4)

                    # Convert RGB to BGR for OpenCV
                    pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

                    video_writer.write(pixels_bgr)
                    frame_count += 1
                    last_frame_time = sim.data.time
                    print(f"Captured frame {frame_count} at t={sim.data.time:.3f}s")

                    if frame_count % 60 == 0:
                        video_writer.release()
                        print(f"Video saved to {video_path}")

                        video_index += 1
                        video_path = f"cable_simulation_{video_index}.mp4"
                        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_x, frame_y))

                viewer.sync()

        video_writer.release()
        print(f"Video saved to {video_path}")

        """