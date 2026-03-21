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


def load_mesh(path: Path):
    mesh = trimesh.load(path, force="mesh")
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.apply_translation(-mesh.center_mass)
    temp_export_file = path.parent / ("._Repositioned_" + path.name)
    mesh.export(temp_export_file)

    return temp_export_file


class Simulation:
    def __init__(self, sat_omega: float = 1, sat_rotation_axis: tuple = (1, 0, 0.3),
                 sat_attach_pos: tuple = (-1.8, 0, -2), av_init_pos: tuple = (-4.8, 10, 1),
                 thruster_react_func = lambda torque_vec: -1 * torque_vec, cable_tension: float = 1,
                 cable_stiffness: float = 7000, cable_damping: float = 30, cable_pt_mass: float = 0.000259,
                 cable_friction: tuple = (0.5, 0.01, 0.01), cable_seg_len: float = 0.3, max_seg_num: int = 200,
                 time_step: float = 0.0001, imp_ratio: float = 2):  # time_step: float = 0.000005

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
        self.cable_node_visual_radius = 0.04  # Visual radius of nodes

        # Calculate the equilibrium length of the string
        equilibrium_cable_stretch = self.cable_tension / self.cable_stiffness  # 0.001m
        self.seg_equilibrium_len = self.cable_seg_len + equilibrium_cable_stretch  # 0.301m


        # Creates the model
        self.xml, self.num_init_segments, anchor_av_init_dir = self._create_model_xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml, None)
        self.data = mujoco.MjData(self.model)

        # Save the IDs of key elements of the model
        self.cylinder_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_rotation")
        self.av_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "av_body")
        # self.av_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "av_free_joint")

        # Get IDs for key nodes and joints in the simulation
        self.cable_body_ids = []
        self.cable_geom_ids = []
        self.cable_tendon_ids = []
        self.cable_jnt_ids = []
        for i in range(self.max_seg_num):
            self.cable_body_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"cable_{i}"))
            self.cable_geom_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"cable_geom_{i}"))
            self.cable_jnt_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cable_free_{i}"))

            if i < self.max_seg_num - 1:
                self.cable_tendon_ids.append(
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, f"cable_tendon_{i}"))

        # Give bodies the proper initial conditions
        qvel_addr = self.model.jnt_dofadr[self.cylinder_jnt_id]
        sat_rot_dir = self.sat_rotation_axis / np.linalg.norm(self.sat_rotation_axis)
        self.data.qvel[qvel_addr + 3:qvel_addr + 6] = self.sat_omega * sat_rot_dir

        # The anchor moves with the cylinder rotation
        init_vel_anchor = np.cross(self.sat_omega * sat_rot_dir, self.sat_attach_pos)

        # All cable nodes should have the same velocity component along the cable
        # This is the velocity that keeps the cable moving as a unit
        cable_direction = anchor_av_init_dir  # Direction along the cable

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
        for i in range(self.num_init_segments - 1, self.max_seg_num - 1):
            tendon_id = self.cable_tendon_ids[i]
            self.model.tendon_stiffness[tendon_id] = 0
            self.model.tendon_damping[tendon_id] = 0

        # Store the state of the cable unspooling
        self.spool_idx = self.num_init_segments - 1
        self.active_count = self.num_init_segments

        # Tension tracking
        self.tension_history = []
        self.time_history = []

    def _create_model_xml(self):
        # Vector from attachment toward origin
        anchor_av_init_dir = self.av_init_pos - self.sat_attach_pos
        init_cable_max_len = np.linalg.norm(anchor_av_init_dir)
        anchor_av_init_dir /= init_cable_max_len

        # Calculate number of segments so spool is just a little bit in front of the origin. This ensures the free
        # link is created in the right direction and has some (short) length
        max_len_to_chain = init_cable_max_len - self.seg_equilibrium_len * 0.1
        init_segments = int(max_len_to_chain / self.seg_equilibrium_len) + 1  # +1 because of the sign post problem

        # Calculate positions for each node at equilibrium spacing
        active_positions = []
        for i in range(0, init_segments):
            pos = self.sat_attach_pos + anchor_av_init_dir * (i * self.seg_equilibrium_len)
            active_positions.append(pos)

        # Build cable bodies with sites
        cable_bodies = ""
        cable_tendons = ""
        for i in range(self.max_seg_num):
            if i < init_segments:
                pos = active_positions[i]
                color = "1 1 0 0"  # Invisible — cable rendered by render_cable()
            else:
                pos = self.av_init_pos + ((i - init_segments) * self.seg_equilibrium_len) * anchor_av_init_dir
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

            if i < self.max_seg_num - 1:  # There are only MAX_SEGMENTS - 1 tendons, again, because of the signpost problem
                cable_tendons += f"""
                <spatial name="cable_tendon_{i}" springlength="{self.cable_seg_len}" 
                        stiffness="{self.cable_stiffness}" damping="{self.cable_damping}" limited="false">
                    <site site="cable_site_{i}"/>
                    <site site="cable_site_{i + 1}"/>
                </spatial>"""

        # Load the mesh files, taking care to reposition the bodies to the origin
        cwd = Path.cwd()
        # sat_obj_file = load_mesh(cwd / "3DModels" / "SatelliteNGPayload.obj")
        sat_obj_file = load_mesh(cwd / "3DModels" / "OffsetCMSatellite_NGPayload.obj")
        av_obj_file = load_mesh(cwd / "3DModels" / "AV_1_0p5_0p5.obj")
        background_im_file = str(cwd / "3DModels" / "Earth5.png")

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

                    <site name="attachment_point" pos="{self.sat_attach_pos[0]} {self.sat_attach_pos[1]} 
                            {self.sat_attach_pos[2]}" size="0.05" rgba="1 0.5 0 1"/>
                    <body name="attachment_body" pos="{self.sat_attach_pos[0]} {self.sat_attach_pos[1]} 
                            {self.sat_attach_pos[2]}">
                        <geom name="attachment_geom" type="sphere" size="0.01" mass="0.001" contype="0" conaffinity="0" rgba="0 0 0 0"/>
                    </body>
                </body>

            {cable_bodies}

            </worldbody>

            <tendon>
                {cable_tendons}
            </tendon>

            <equality>
                <connect name="cable_attachment" body1="cable_0" body2="attachment_body" 
                         anchor="0 0 0" solref="0.0005 1" solimp="0.99 0.999 0.0001"/>
            </equality>
        </mujoco>
        """

        return xml, init_segments, anchor_av_init_dir

    def _get_pos(self, idx):
        return self.data.xpos[self.cable_body_ids[idx]].copy()

    def step(self):
        # Apply tension force on spool - must be done before mj_step
        self._apply_tension_and_forces()

        # Step forward
        mujoco.mj_step(self.model, self.data)

        self._maybe_spawn()
        self._maybe_despawn()
        self._record_tension()

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
        # STEP 1: Apply tension to the rope (and the corresponding opposing force on the AV)
        spool_pos = self._get_pos(self.spool_idx)
        body_id = self.cable_body_ids[self.spool_idx]

        # Direction from spool TOWARD origin (along the free link)
        # This pulls the spool away from the chain/cylinder, creating tension
        toward_origin = self._get_av_pos() - spool_pos
        dist = np.linalg.norm(toward_origin)

        # if dist > 1e-6:
        direction = toward_origin / dist
        force = direction * self.cable_tension

        # Clear any existing applied forces first
        self.data.xfrc_applied[body_id, :] = 0

        # Use mj_applyFT to properly apply force and torque
        torque_vec = np.zeros(3, dtype=np.float64)

        # Create array to receive the generalized forces
        # Applies the forces to the rope
        qfrc_applied = np.zeros(self.model.nv, dtype=np.float64)
        mujoco.mj_applyFT(self.model, self.data, force, torque_vec, spool_pos, body_id, qfrc_applied)

        # Now apply the reaction forces on the AV
        thruster_force = self.thruster_react_func(force)
        self.current_thruster_force = thruster_force.copy()
        mujoco.mj_applyFT(self.model, self.data, thruster_force, torque_vec,
                          self._get_av_pos(), self.av_body_id, qfrc_applied)
        self.data.qfrc_applied[:] = qfrc_applied  # Note this adds instead of re

    def _get_av_pos(self):
        # start_addr = self.model.jnt_dofadr[self.av_joint_id]
        # return self.data.qpos[start_addr:start_addr+3]

        return self.av_init_pos[0], self.av_init_pos[1], self.av_init_pos[2]

    def _maybe_spawn(self):
        if self.active_count < self.max_seg_num:
            spool_pos = self._get_pos(self.spool_idx)

            # Calculate free link length (spool to origin)
            free_link_length = np.linalg.norm(self._get_av_pos() - spool_pos)

            if free_link_length > self.seg_equilibrium_len * 1.1:
                print(f"\n=== SPAWN TRIGGER t={self.data.time:.4f}s ===")
                print(f"Free link length: {free_link_length:.4f}m > threshold {self.seg_equilibrium_len:.4f}m")
                self._spawn_segment()

    def _spawn_segment(self):
        # Forward the physics state. This must be done at the start AND at the end to get the proper location for the
        # old spool
        mujoco.mj_forward(self.model, self.data)

        new_idx = self.active_count
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
        self.model.geom_rgba[self.cable_geom_ids[new_idx]] = [0, 1, 0, 0]
        self.model.geom_rgba[self.cable_geom_ids[old_spool]] = [1, 1, 0, 0]

        # Update spool index and active count
        self.spool_idx = new_idx
        self.active_count = new_idx + 1

        # Now that the new spool is spawned, forward the state again. Forwarding twice is required to reduce tension
        # spikes
        mujoco.mj_forward(self.model, self.data)

    def _maybe_despawn(self):
        spool_pos = self._get_pos(self.spool_idx)

        # Calculate free link length (spool to AV)
        free_link_length = np.linalg.norm(self._get_av_pos() - spool_pos)

        # Despawn threshold - when free link is very short, absorb the spool back
        despawn_threshold = self.seg_equilibrium_len * 0.1  # 10% of equilibrium length

        if free_link_length < despawn_threshold:
            print(f"\n=== DESPAWN TRIGGER t={self.data.time:.4f}s ===")
            print(f"Free link length: {free_link_length:.4f}m < threshold {despawn_threshold:.4f}m")
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

        # Move the old spool segment back to inactive position (behind AV)
        old_spool_jnt_id = self.cable_jnt_ids[old_spool]
        qpos_adr = self.model.jnt_qposadr[old_spool_jnt_id]
        qvel_adr = self.model.jnt_dofadr[old_spool_jnt_id]

        # Calculate inactive position (behind AV along cable direction)
        av_pos = self._get_av_pos()
        new_spool_pos = self._get_pos(new_spool)
        away_from_sat = av_pos - new_spool_pos
        away_from_sat_dir = away_from_sat / np.linalg.norm(away_from_sat)

        inactive_pos = av_pos + away_from_sat_dir * (
                    (old_spool - self.num_init_segments + 1) * self.seg_equilibrium_len)

        # Set position and zero velocity
        self.data.qpos[qpos_adr:qpos_adr + 3] = inactive_pos
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0

        # Update visual colors (keep invisible — cable rendered by render_cable)
        # TODO parameterize the colors
        self.model.geom_rgba[self.cable_geom_ids[old_spool]] = [0.5, 0.5, 0.5, 0]
        self.model.geom_rgba[self.cable_geom_ids[new_spool]] = [0, 1, 0, 0]

        # Update spool index and active count
        self.spool_idx = new_spool
        self.active_count = old_spool  # active_count is now old_spool (not old_spool + 1)

        print(f"  new spool_idx: {self.spool_idx}, active_count: {self.active_count}")

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
        fovy = np.radians(self.model.vis.global_.fovy)
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

    def render_cable(self, scene):
        spool_pos = self._get_pos(self.spool_idx)

        g = scene.geoms[scene.ngeom]
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, self.cable_visual_radius, self._get_av_pos(), spool_pos)
        g.rgba[:] = [1.0, 0.4, 0.15, 1.0]
        g.label = ""  # Clear stale label from previous frame
        scene.ngeom += 1

        for i in range(self.spool_idx):
            p1 = self._get_pos(i)
            p2 = self._get_pos(i + 1)

            g = scene.geoms[scene.ngeom]
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, self.cable_visual_radius, p1, p2)
            g.rgba[:] = [1.0, 0.45, 0.15, 1.0]
            g.label = ""  # Clear stale label from previous frame
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
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            # viewer.cam.azimuth = 90
            # viewer.cam.elevation = -20
            # viewer.cam.distance = 25
            # viewer.cam.lookat[:] = [0, 0, 0]
            viewer.cam.azimuth = -0.8289683948863515
            viewer.cam.elevation = -21.25310724431816
            viewer.cam.distance = 34.68901593838663
            viewer.cam.lookat[:] = [0, 5, 0]

            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = view_tendons
            sleep(10)

            while viewer.is_running():
                sim.step()

                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    sim.render_cable(viewer.user_scn)
                    sim.render_overlay(viewer.user_scn, viewer)

                


                if sim.data.time > 5 and not hasattr(sim, 'plotted'):
                    sim.plot_tension()
                    sim.plotted = True

                if sim.data.time > 480:
                    exit()
                # if sim.data.time > 0.5:
                #     sim = all_simulations[sim_num]
                #     sim_num += 1

                #     with viewer.lock():
                #         viewer.m = sim.model
                #         viewer.d = sim.data

                #     viewer.sync()


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