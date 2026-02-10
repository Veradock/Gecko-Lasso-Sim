# TODO: clean up the code, clean up the visualization, and record!!
"""
MuJoCo Cable Wrapping Simulation - Native Physics Version

A cable unspools from a fixed point and wraps around a spinning cylinder.
Uses MuJoCo's native contact and friction calculations.

Requirements:
    pip install mujoco numpy matplotlib

Run:
    python cable_wrapping_sim.py
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import trimesh

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
CYLINDER_OMEGA = 1       # rad/s rotation speed

CABLE_SEGMENT_LENGTH = 0.3  # Length of each cable segment
CABLE_RADIUS = 0.03         # Physical radius of cable
NODE_RADIUS = 0.04          # Visual radius of nodes

SAT_ATTACH_PT = np.array([-1.8, 0, -1])
AV_INIT_POS = np.array([-1.8, 2 * 1.8, 0])

MAX_SEGMENTS = 200           # Maximum number of pre-allocated cable segments

# Cable physical properties
CABLE_MASS = 0.05           # Mass per segment (kg)

# Friction coefficients (MuJoCo format: sliding, torsional, rolling)
CABLE_FRICTION = [0.5, 0.01, 0.01]

# Build spatial tendons connecting adjacent segments
# Only initially active tendons are between active segments
SPRING_STIFFNESS = 50000.0  # N/m - stiff spring

# Use an overdamped system (5x critical damping)
SPRING_DAMPING = 5 * 2 * np.sqrt(CABLE_MASS * SPRING_STIFFNESS)  # Ns/m
TIME_STEP = 0.00005
IMP_RATIO = 2
SAT_ROT_AX = np.array((0, 1, 1))

# For equilibrium with 10N tension and k=10000 N/m:
SPOOL_TENSION = 10
EQUILIBRIUM_STRETCH_PER_LINK = SPOOL_TENSION / SPRING_STIFFNESS  # 0.001m
EQUILIBRIUM_LINK_LENGTH = CABLE_SEGMENT_LENGTH + EQUILIBRIUM_STRETCH_PER_LINK  # 0.301m

def load_mesh(path: Path):
    # Loads a mesh from a .obj file. This function ensures that the origin of the body is placed at the center of mass.
    mesh = trimesh.load(path, force="mesh")
    mesh.apply_translation(-mesh.center_mass)
    temp_export_file = path.parent / ("._Repositioned_" + path.name)
    mesh.export(temp_export_file)
    return temp_export_file


def create_model_xml():
    # Vector from attachment toward origin
    anchor_av_init_dir = AV_INIT_POS - SAT_ATTACH_PT
    init_cable_max_len = np.linalg.norm(anchor_av_init_dir)
    anchor_av_init_dir /= init_cable_max_len

    # Calculate number of segments so spool is just a little bit in front of the origin. This ensures the free
    # link is created in the right direction and has some (short) length
    max_len_to_chain = init_cable_max_len - EQUILIBRIUM_LINK_LENGTH * 0.1
    init_segments = int(max_len_to_chain / EQUILIBRIUM_LINK_LENGTH) + 1  # +1 because of the sign post problem

    # Calculate positions for each node at equilibrium spacing
    active_positions = []
    for i in range(0, init_segments):
        pos = SAT_ATTACH_PT + anchor_av_init_dir * (i * EQUILIBRIUM_LINK_LENGTH)
        active_positions.append(pos)

    # Spool is at the end of the chain - free link goes from spool to origin
    # print(f"DEBUG: attach_to_origin={attach_to_origin:.4f}m")
    # print(f"DEBUG: init_segments={init_segments}")
    # print(f"DEBUG: equilibrium link length={EQUILIBRIUM_LINK_LENGTH:.4f}m")
    # print(f"DEBUG: spool pos={spool_pos}")
    # print(f"DEBUG: free link length={free_link_length:.4f}m")

    # Build cable bodies with sites
    cable_bodies = ""
    cable_tendons = ""
    for i in range(MAX_SEGMENTS):
        if i < init_segments:
            pos = active_positions[i]
            if i == 0:
                color = "1 0.5 0 1"
            elif i == init_segments - 1:
                color = "0 1 0 1"
            else:
                color = "1 1 0 1"
        else:
            # This must go off of AV_INIT_POS since the length of the free link is not prescribed
            pos = AV_INIT_POS + ((i - init_segments) * EQUILIBRIUM_LINK_LENGTH) * anchor_av_init_dir
            color = "0.5 0.5 0.5 1"

        cable_bodies += f"""
        <body name="cable_{i}" pos="{pos[0]} {pos[1]} {pos[2]}">
            <freejoint name="cable_free_{i}"/>
            <geom name="cable_geom_{i}" type="sphere" size="{NODE_RADIUS}" mass="{CABLE_MASS}"
                  friction="{CABLE_FRICTION[0]} {CABLE_FRICTION[1]} {CABLE_FRICTION[2]}"
                  rgba="{color}" condim="6" contype="1" conaffinity="2"/>
            <site name="cable_site_{i}" pos="0 0 0" size="0.005"/>
        </body>
        """

        if i < MAX_SEGMENTS - 1:  # There are only MAX_SEGMENTS - 1 tendons, again, because of the signpost problem
            cable_tendons += f"""
            <spatial name="cable_tendon_{i}" springlength="{CABLE_SEGMENT_LENGTH}" stiffness="{SPRING_STIFFNESS}" damping="{SPRING_DAMPING}" limited="false">
                <site site="cable_site_{i}"/>
                <site site="cable_site_{i+1}"/>
            </spatial>"""

    # Load the mesh files, taking care to reposition the bodies to the origin
    cwd = Path.cwd()
    sat_obj_file = load_mesh(cwd / "3DModels" / "SatelliteNGPayload.obj")
    av_obj_file = load_mesh(cwd / "3DModels" / "AV_1_0p5_0p5.obj")

    # Next build the main XML defining the simulation
    xml = f"""
    <mujoco model="Gecko Lasso Simulation">
        <option gravity="0 0 0" timestep="{TIME_STEP}" integrator="implicit" cone="elliptic" impratio="{IMP_RATIO}"/>
        
        <visual>
            <global offwidth="1920" offheight="1080"/>
            <quality shadowsize="4096"/>
        </visual>
        
        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.1 0.1 0.2" rgb2="0.3 0.3 0.4" width="512" height="512"/>
            <material name="cylinder_mat" rgba="0.3 0.5 0.8 0.8"/>
            <material name="ground_mat" rgba="0.2 0.2 0.2 1" reflectance="0.1"/>
            <mesh name="sat_mesh" file="{sat_obj_file}"/>
            <mesh name="av_mesh" file="{av_obj_file}"/>
        </asset>
        
        <worldbody>
            <geom type="plane" size="20 20 0.1" pos="0 0 -6" material="ground_mat" contype="0" conaffinity="0"/>
            
             <body name="av_body" pos="{AV_INIT_POS[0]} {AV_INIT_POS[1]} {AV_INIT_POS[2]}">
                <freejoint name="av_free_joint"/>
                <geom type="mesh" mesh="av_mesh" contype="4" conaffinity="2" friction="0 0 0" condim="6" rgba="0.2 0.2 0.9 1"/>
                <site name="cable_origin" pos="0 0 0" size="0.06" rgba="0 1 0 1"/>
            </body>

            
            <body name="cylinder" pos="0 0 0">
                <!-- <joint name="cylinder_rotation" type="hinge" axis="{SAT_ROT_AX[0]} {SAT_ROT_AX[1]} {SAT_ROT_AX[2]}" damping="0"/> -->
                <freejoint name="cylinder_rotation"/>
                <geom type="mesh" mesh="sat_mesh" contype="2" conaffinity="1" density="500000" 
                    friction="{CABLE_FRICTION[0]} {CABLE_FRICTION[1]} {CABLE_FRICTION[2]}" condim="6" rgba="0.9 0.2 0.2 1"/>
                
                <site name="attachment_point" pos="{SAT_ATTACH_PT[0]} {SAT_ATTACH_PT[1]} {SAT_ATTACH_PT[2]}" size="0.05" rgba="1 0.5 0 1"/>
                <body name="attachment_body" pos="{SAT_ATTACH_PT[0]} {SAT_ATTACH_PT[1]} {SAT_ATTACH_PT[2]}">
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
        
        <!--actuator>
            <velocity name="cylinder_motor" joint="cylinder_rotation" kv="10000"/>
        </actuator-->
    </mujoco>
    """

    return xml, init_segments, anchor_av_init_dir


def get_thruster_reaction():
    return (-10, 0, 0)

class Simulation:
    def __init__(self):
        # Creates the model
        self.xml, self.num_init_segments, anchor_av_init_dir = create_model_xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml, None)
        self.data = mujoco.MjData(self.model)

        # Save the IDs of key elements of the model
        self.cylinder_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_rotation")
        self.av_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "av_body")
        self.av_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "av_free_joint")

        # self.motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cylinder_motor")

        # Give bodies the proper initial conditions
        qvel_addr = self.model.jnt_dofadr[self.cylinder_jnt_id]
        sat_rot_dir = SAT_ROT_AX / np.linalg.norm(SAT_ROT_AX)
        self.data.qvel[qvel_addr+3:qvel_addr+6] = CYLINDER_OMEGA * sat_rot_dir

        # Move onto the nodes on the cables
        init_vel_anchor = np.cross(SAT_ATTACH_PT, CYLINDER_OMEGA * sat_rot_dir)
        init_vel_cable = -1 * anchor_av_init_dir * np.linalg.norm(init_vel_anchor)

        # Applies the velocity at the attachment point
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cable_free_0")
        qvel_addr = self.model.jnt_dofadr[body_id]
        self.data.qvel[qvel_addr:qvel_addr + 3] = init_vel_anchor

        # Applies the initial velocities along the cable
        for string_node in range(1, self.num_init_segments - 1):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cable_free_{string_node}")
            qvel_addr = self.model.jnt_dofadr[body_id]

            self.data.qvel[qvel_addr:qvel_addr + 3] = init_vel_cable

        # Forward pass to initialize physics state consistently
        mujoco.mj_forward(self.model, self.data)
        # mujoco.mj_subtreeVel(self.model, self.data)

        self.cable_body_ids = []
        self.cable_geom_ids = []
        self.cable_tendon_ids = []
        self.cable_jnt_ids = []
        for i in range(MAX_SEGMENTS):
            self.cable_body_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"cable_{i}"))
            self.cable_geom_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"cable_geom_{i}"))
            self.cable_jnt_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cable_free_{i}"))

            if i < MAX_SEGMENTS - 1:
                self.cable_tendon_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, f"cable_tendon_{i}"))

        # Deactivate tendons for inactive segments by setting stiffness to 0
        for i in range(self.num_init_segments - 1, MAX_SEGMENTS - 1):
            tendon_id = self.cable_tendon_ids[i]
            self.model.tendon_stiffness[tendon_id] = 0
            self.model.tendon_damping[tendon_id] = 0

        # Store the state of the cable unspooling
        self.spool_idx = self.num_init_segments - 1
        self.active_count = self.num_init_segments

        # Tension tracking
        self.tension_history = []
        self.time_history = []

    def _get_pos(self, idx):
        return self.data.xpos[self.cable_body_ids[idx]].copy()

    def step(self):
        # Apply tension force on spool - must be done before mj_step
        self._apply_tension_and_forces()

        # Step forward
        mujoco.mj_step(self.model, self.data)

        self._maybe_spawn()
        self._record_tension()

    def _record_tension(self):
        self.time_history.append(self.data.time)

        """
        # Debug: print detailed info at specific times
        debug_times = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]
        # should_debug = any(abs(self.time - t) < 0.003 for t in debug_times)
        should_debug = False

        if should_debug:
            print(f"\n=== TENSION DEBUG t={self.time:.3f}s ===")
            print(f"active_count={self.active_count}, spool_idx={self.spool_idx}")

            # Check spool position
            spool_pos = self._get_pos(self.spool_idx)
            print(f"spool pos: {spool_pos}")
            # print(f"distance to origin: {np.linalg.norm(CABLE_ORIGIN - spool_pos):.4f}m")

            # Check last few tendon tensions
            print("Tendon tensions (last 5 active):")
            for i in range(max(0, self.active_count - 6), self.active_count - 1):
                tendon_id = self.cable_tendon_ids[i]
                length = self.data.ten_length[tendon_id]
                stiffness = self.model.tendon_stiffness[tendon_id]
                stretch = length - CABLE_SEGMENT_LENGTH
                force = stiffness * stretch
                print(f"  tendon {i}: length={length:.4f}, stretch={stretch:.6f}, force={force:.2f}N")
        """

        # Get tendon forces
        tensions = []
        for i in range(self.active_count - 1):
            if i < len(self.cable_tendon_ids):
                tendon_id = self.cable_tendon_ids[i]
                length = self.data.ten_length[tendon_id]
                stretch = length - CABLE_SEGMENT_LENGTH
                tensions.append(SPRING_STIFFNESS * stretch)
            else:
                tensions.append(0)

        # Add free link tension
        tensions.append(SPOOL_TENSION)

        self.tension_history.append(tensions)

    def _apply_tension_and_forces(self):
        # STEP 1: Apply tension to the rope (and the corresponding opposing force on the AV)
        spool_pos = self._get_pos(self.spool_idx)
        body_id = self.cable_body_ids[self.spool_idx]

        # Direction from spool TOWARD origin (along the free link)
        # This pulls the spool away from the chain/cylinder, creating tension
        toward_origin = self._get_AV_pos() - spool_pos
        dist = np.linalg.norm(toward_origin)

        # if dist > 1e-6:
        direction = toward_origin / dist
        force = direction * SPOOL_TENSION

        # Clear any existing applied forces first
        self.data.xfrc_applied[body_id, :] = 0

        # Use mj_applyFT to properly apply force and torque
        # force_vec = np.array(force, dtype=np.float64)
        torque_vec = np.zeros(3, dtype=np.float64)
        # point = np.array(spool_pos, dtype=np.float64)

        # Create array to receive the generalized forces
        # Applies the forces to the rope
        qfrc_applied = np.zeros(self.model.nv, dtype=np.float64)
        mujoco.mj_applyFT(self.model, self.data, force, torque_vec, spool_pos, body_id, qfrc_applied)

        # Now apply the reaction forces on the AV
        mujoco.mj_applyFT(self.model, self.data, -1 * force + get_thruster_reaction(), torque_vec,
                          self._get_AV_pos(), self.av_body_id, qfrc_applied)
        self.data.qfrc_applied[:] = qfrc_applied  # Note this adds instead of re

    '''
    def _debug_force_balance(self):
        """Debug the force balance on the spool body."""
        spool_body_id = self.cable_body_ids[self.spool_idx]
        spool_pos = self._get_pos(self.spool_idx)

        # Get the joint/dof indices for the spool's freejoint
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cable_free_{self.spool_idx}")
        qpos_adr = self.model.jnt_qposadr[joint_id]
        qvel_adr = self.model.jnt_dofadr[joint_id]

        # For a freejoint, DOFs are: 3 angular + 3 linear velocity
        lin_dof_start = qvel_adr + 3

        print(f"\n=== FORCE BALANCE DEBUG t={self.time:.4f}s ===")
        print(f"Spool idx: {self.spool_idx}, body_id: {spool_body_id}")
        print(f"Spool position: {spool_pos}")

        # Applied external force
        applied_force = self.data.qfrc_applied[lin_dof_start:lin_dof_start+3]
        print(f"qfrc_applied (linear DOFs): {applied_force}")

        # Also check xfrc_applied
        xfrc = self.data.xfrc_applied[spool_body_id, 3:6]
        print(f"xfrc_applied (force): {xfrc} magnitude: {np.linalg.norm(xfrc):.4f}N")

        # Tendon connecting spool to previous segment
        tendon_idx = self.spool_idx - 1
        tendon_id = self.cable_tendon_ids[tendon_idx]
        tendon_length = self.data.ten_length[tendon_id]
        tendon_velocity = self.data.ten_velocity[tendon_id]
        tendon_stiffness = self.model.tendon_stiffness[tendon_id]
        tendon_damping = self.model.tendon_damping[tendon_id]

        stretch = tendon_length - CABLE_SEGMENT_LENGTH
        spring_force = tendon_stiffness * stretch
        damping_force = tendon_damping * tendon_velocity
        total_tendon_force = spring_force + damping_force

        print(f"\nTendon {tendon_idx} (spool to prev):")
        print(f"  length: {tendon_length:.6f}m (rest: {CABLE_SEGMENT_LENGTH}m)")
        print(f"  stretch: {stretch:.6f}m")
        print(f"  velocity: {tendon_velocity:.6f}m/s")
        print(f"  spring force: {spring_force:.4f}N (k={tendon_stiffness})")
        print(f"  damping force: {damping_force:.4f}N (c={tendon_damping})")
        print(f"  total tendon force (magnitude): {total_tendon_force:.4f}N")

        # Direction of tendon force on spool
        prev_pos = self._get_pos(self.spool_idx - 1)
        toward_prev = prev_pos - spool_pos
        toward_prev_dir = toward_prev / np.linalg.norm(toward_prev)
        tendon_force_vec = toward_prev_dir * total_tendon_force
        print(f"  direction (toward prev): {toward_prev_dir}")
        print(f"  tendon force vector: {tendon_force_vec}")

        # Net force estimate
        net_force_estimate = applied_force + tendon_force_vec
        print(f"\nNet force estimate (qfrc_applied + tendon): {net_force_estimate}")
        print(f"Net force magnitude: {np.linalg.norm(net_force_estimate):.4f}N")

        # Check all qfrc arrays
        print(f"\n--- BEFORE mj_forward ---")
        print(f"Generalized forces (linear DOFs {lin_dof_start}:{lin_dof_start+3}):")
        print(f"  qfrc_passive: {self.data.qfrc_passive[lin_dof_start:lin_dof_start+3]}")
        print(f"  qfrc_applied: {self.data.qfrc_applied[lin_dof_start:lin_dof_start+3]}")
        print(f"  qfrc_bias: {self.data.qfrc_bias[lin_dof_start:lin_dof_start+3]}")
        print(f"  qfrc_constraint: {self.data.qfrc_constraint[lin_dof_start:lin_dof_start+3]}")
        print(f"  qfrc_actuator: {self.data.qfrc_actuator[lin_dof_start:lin_dof_start+3]}")

        mujoco.mj_forward(self.model, self.data)

        print(f"\n--- AFTER mj_forward ---")
        print(f"Generalized forces (linear DOFs {lin_dof_start}:{lin_dof_start+3}):")
        print(f"  qfrc_passive: {self.data.qfrc_passive[lin_dof_start:lin_dof_start+3]}")
        print(f"  qfrc_applied: {self.data.qfrc_applied[lin_dof_start:lin_dof_start+3]}")
        print(f"  qfrc_bias: {self.data.qfrc_bias[lin_dof_start:lin_dof_start+3]}")
        print(f"  qfrc_constraint: {self.data.qfrc_constraint[lin_dof_start:lin_dof_start+3]}")
        print(f"  qfrc_actuator: {self.data.qfrc_actuator[lin_dof_start:lin_dof_start+3]}")

        # Constraint solver state
        print(f"\n--- CONSTRAINT SOLVER STATE ---")
        print(f"Number of active constraints (nefc): {self.data.nefc}")
        print(f"Number of contacts (ncon): {self.data.ncon}")

        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "cable_attachment")
        print(f"Equality constraint 'cable_attachment' id: {eq_id}")

        if self.data.nefc > 0:
            print(f"efc_force (first 10): {self.data.efc_force[:min(10, self.data.nefc)]}")
            print(f"efc_type (first 10): {self.data.efc_type[:min(10, self.data.nefc)]}")

        # Tendon forces
        print(f"\n--- TENDON FORCES FROM MUJOCO ---")
        for i in range(max(0, self.spool_idx - 3), self.spool_idx):
            tid = self.cable_tendon_ids[i]
            tlen = self.data.ten_length[tid]
            tvel = self.data.ten_velocity[tid]
            print(f"  Tendon {i}: len={tlen:.6f}, vel={tvel:.6f}, stiff={self.model.tendon_stiffness[tid]}, damp={self.model.tendon_damping[tid]}")

        # xfrc_applied check
        print(f"\n--- xfrc_applied CHECK ---")
        nonzero_xfrc = []
        for i in range(self.model.nbody):
            force = self.data.xfrc_applied[i, 3:6]
            if np.linalg.norm(force) > 1e-10:
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                nonzero_xfrc.append((i, body_name, force))
        print(f"Bodies with nonzero xfrc_applied: {nonzero_xfrc}")

        # qfrc_applied check
        print(f"\n--- qfrc_applied CHECK ---")
        nonzero_qfrc = np.where(np.abs(self.data.qfrc_applied) > 1e-10)[0]
        if len(nonzero_qfrc) > 0:
            print(f"Nonzero qfrc_applied indices: {nonzero_qfrc}")
            print(f"Nonzero qfrc_applied values: {self.data.qfrc_applied[nonzero_qfrc]}")
        else:
            print("qfrc_applied is all zeros!")

        # cfrc_ext
        print(f"\n--- cfrc_ext (external forces on bodies) ---")
        spool_cfrc = self.data.cfrc_ext[spool_body_id]
        print(f"Spool cfrc_ext: torque={spool_cfrc[:3]}, force={spool_cfrc[3:]}")

        # Velocity/acceleration
        print(f"\n--- VELOCITY/ACCELERATION ---")
        spool_vel = self.data.qvel[lin_dof_start:lin_dof_start+3]
        spool_acc = self.data.qacc[lin_dof_start:lin_dof_start+3]
        print(f"Spool linear velocity: {spool_vel}")
        print(f"Spool linear acceleration: {spool_acc}")

        if hasattr(self.data, 'qacc_warmstart'):
            print(f"qacc_warmstart: {self.data.qacc_warmstart[lin_dof_start:lin_dof_start+3]}")

        mass = CABLE_MASS
        expected_acc = net_force_estimate / mass
        print(f"Expected acceleration (F/m): {expected_acc}")

        observed_force = spool_acc * mass
        print(f"Force implied by observed acceleration: {observed_force}")
        print(f"Magnitude of implied force: {np.linalg.norm(observed_force):.2f}N")

        print(f"\n--- BODY PROPERTIES ---")
        print(f"Spool body mass: {self.model.body_mass[spool_body_id]}")
        print(f"Spool body inertia: {self.model.body_inertia[spool_body_id]}")

    def _debug_first_spawn(self):
        # Debug first spawn
        if self.debug_first_spawn_time is not None:
            elapsed = self.data.time - self.debug_first_spawn_time
            if elapsed > 0.5:
                self.debug_first_spawn_time = None
            elif int(elapsed * 50) != int((elapsed - self.model.opt.timestep) * 50):
                idx = self.debug_first_spawn_new_idx
                pos = self._get_pos(idx)
                vel = self.data.cvel[self.cable_body_ids[idx], 3:6]
                dist_from_origin = pos[0] - CABLE_ORIGIN[0]
                print(f"  SPAWN1 t+{elapsed:.3f}s: pos_x={pos[0]:.4f}, dist={dist_from_origin:.4f}, vel_x={vel[0]:.4f}")

        # Debug second spawn
        if self.debug_second_spawn_time is not None:
            elapsed = self.data.time - self.debug_second_spawn_time
            if elapsed > 0.5:
                self.debug_second_spawn_time = None
            elif int(elapsed * 50) != int((elapsed - self.model.opt.timestep) * 50):
                idx = self.debug_second_spawn_new_idx
                pos = self._get_pos(idx)
                vel = self.data.cvel[self.cable_body_ids[idx], 3:6]
                dist_from_origin = pos[0] - CABLE_ORIGIN[0]
                print(f"  SPAWN2 t+{elapsed:.3f}s: pos_x={pos[0]:.4f}, dist={dist_from_origin:.4f}, vel_x={vel[0]:.4f}")
    '''

    def _get_AV_pos(self):
        start_addr = self.model.jnt_dofadr[self.av_joint_id]
        return self.data.qpos[start_addr:start_addr+3]

    def _maybe_spawn(self):
        if self.active_count < MAX_SEGMENTS:
            spool_pos = self._get_pos(self.spool_idx)

            # Calculate free link length (spool to origin)
            free_link_length = np.linalg.norm(self._get_AV_pos() - spool_pos)

            if free_link_length > EQUILIBRIUM_LINK_LENGTH:
                print(f"\n=== SPAWN TRIGGER t={self.data.time:.4f}s ===")
                print(f"Free link length: {free_link_length:.4f}m > threshold {EQUILIBRIUM_LINK_LENGTH:.4f}m")
                self._spawn_segment()

    def _spawn_segment(self):
        # Forward the physics state. This must be done at the start AND at the end to get the proper location for the
        # old spool
        mujoco.mj_forward(self.model, self.data)

        new_idx = self.active_count
        old_spool = self.spool_idx
        # spawn_number = new_idx - self.init_segments + 1

        old_spool_pos = self._get_pos(old_spool)

        # Get old spool velocity from qvel
        old_joint_id = self.cable_jnt_ids[old_spool]
        old_qvel_adr = self.model.jnt_dofadr[old_joint_id]
        # old_spool_vel = self.data.qvel[old_qvel_adr:old_qvel_adr+3].copy()

        # Calculate current free link length
        # free_link_length = np.linalg.norm(CABLE_ORIGIN - old_spool_pos)

        # print(f"\nSPAWN #{spawn_number}: new_idx={new_idx}, old_spool={old_spool}")
        # print(f"  old_spool pos: {old_spool_pos}")
        # print(f"  old_spool vel: {old_spool_vel}")
        # print(f"  free link length at spawn: {free_link_length:.6f}m")

        # Get new segment's joint addresses
        qpos_adr = self.model.jnt_qposadr[self.cable_jnt_ids[new_idx]]
        qvel_adr = self.model.jnt_dofadr[self.cable_jnt_ids[new_idx]]

        # Direction from old spool toward origin
        toward_origin = self._get_AV_pos() - old_spool_pos
        toward_origin_dir = toward_origin / np.linalg.norm(toward_origin)

        # Position new spool at equilibrium distance from old spool, toward origin
        spawn_pos = old_spool_pos + toward_origin_dir * EQUILIBRIUM_LINK_LENGTH

        # new_free_link = np.linalg.norm(CABLE_ORIGIN - spawn_pos)

        # Debug: check tendon length that will result
        # tendon_length_will_be = np.linalg.norm(spawn_pos - old_spool_pos)

        # print(f"  spawn pos: {spawn_pos}")
        # rint(f"  new free link length: {new_free_link:.6f}m")
        # print(f"  tendon length (old_spool to new): {tendon_length_will_be:.6f}m")
        # print(f"  tendon rest length: {CABLE_SEGMENT_LENGTH}m")
        # print(f"  tendon initial stretch: {tendon_length_will_be - CABLE_SEGMENT_LENGTH:.6f}m")
        # print(f"  tendon initial force: {(tendon_length_will_be - CABLE_SEGMENT_LENGTH) * SPRING_STIFFNESS:.2f}N")

        # Set new segment position and orientation
        self.data.qpos[qpos_adr:qpos_adr+3] = spawn_pos
        self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
        print("ASSIGNING THE NEW SPOOL A POSITION OF " + str(spawn_pos))

        # Set velocity to zero - the new spool emerges from the fixed origin point
        # The free link represents cable being paid out, so the spool end starts stationary
        # print(f"  assigning velocity: [0, 0, 0] (stationary - emerging from fixed origin)")
        # self.data.qvel[qvel_adr:qvel_adr+3] = [0, 0, 0]
        # self.data.qvel[qvel_adr+3:qvel_adr+6] = 0

        # To result in smoother spawning, we will give the cable an initial velocity
        self.data.qvel[qvel_adr:qvel_adr+3] = self.data.qvel[old_qvel_adr:old_qvel_adr+3]

        # Activate tendon between old spool and new segment
        tendon_id = self.cable_tendon_ids[old_spool]
        print(f"  activating tendon {old_spool} (connects seg {old_spool} to seg {new_idx})")
        self.model.tendon_stiffness[tendon_id] = SPRING_STIFFNESS
        self.model.tendon_damping[tendon_id] = SPRING_DAMPING

        # Update visual colors
        self.model.geom_rgba[self.cable_geom_ids[new_idx]] = [0, 1, 0, 1]
        self.model.geom_rgba[self.cable_geom_ids[old_spool]] = [1, 1, 0, 1]

        # Update spool index and active count
        self.spool_idx = new_idx
        self.active_count = new_idx + 1

        # Debug: check state after mj_forward
        # actual_tendon_length = self.data.ten_length[tendon_id]
        # print(f"  POST mj_forward tendon length: {actual_tendon_length:.6f}m")
        # print(f"  POST mj_forward old spool pos: {self._get_pos(old_spool)}")
        # print(f"  new spool_idx: {self.spool_idx}, active_count: {self.active_count}")

        # Now that the new spool is spawned, forward the state again. Forwarding twice is required to reduce tension
        # spikes
        mujoco.mj_forward(self.model, self.data)

    """
    def get_info(self):
        return f"Time: {self.time:.2f}s | Wraps: {self.wrap_count:.2f} | Segments: {self.active_count}"
    """

    def plot_tension(self):
        if not self.tension_history:
            print("No tension data recorded")
            return

        print("\n" + "="*50)
        print("TENSION SUMMARY (Newtons)")
        print("="*50)

        for i, t in enumerate(self.time_history):
            if i == 0 or i == len(self.time_history)-1 or (t % 0.1 < 0.003):
                tensions = self.tension_history[i]
                if len(tensions) > 5:
                    print(f"t={t:.3f}s: last 5 links: {[f'{x:.2f}' for x in tensions[-6:-1]]}, spool: {tensions[-1]:.2f}N")
                else:
                    print(f"t={t:.3f}s: all links: {[f'{x:.2f}' for x in tensions]}")

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

        plt.axhline(y=10.0, color='gray', linestyle='--', label='Applied tension (10N)')

        plt.xlabel('Time (s)')
        plt.ylabel('Tension (N)')
        plt.title('Link Tension Over Time')
        plt.ylim(bottom=0)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def render_cable(self, scene):
        spool_pos = self._get_pos(self.spool_idx)

        g = scene.geoms[scene.ngeom]
        # mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, CABLE_RADIUS,
        #     self._get_AV_pos().reshape(3,1), spool_pos.reshape(3,1))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, CABLE_RADIUS, self._get_AV_pos(), spool_pos)
        g.rgba[:] = [0.7, 0.1, 0.1, 1.0]
        scene.ngeom += 1

        for i in range(self.spool_idx):
            p1 = self._get_pos(i)
            p2 = self._get_pos(i + 1)

            g = scene.geoms[scene.ngeom]
            # mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, CABLE_RADIUS, p1.reshape(3,1), p2.reshape(3,1))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, CABLE_RADIUS, p1, p2)
            g.rgba[:] = [0.9, 0.2, 0.1, 1.0]
            scene.ngeom += 1



def main():
    print("=" * 50)
    print("MuJoCo Cable Wrapping Simulation")
    print("=" * 50)

    sim = Simulation()

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 25
        viewer.cam.lookat[:] = [0, 0, 0]

        frame = 0
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        while viewer.is_running():
            sim.step()

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                sim.render_cable(viewer.user_scn)

            frame += 1
            # if frame % 500 == 0:
            #     print(sim.get_info())

            if sim.data.time > 1 and not hasattr(sim, 'plotted'):
                sim.plot_tension()
                sim.plotted = True

            viewer.sync()

    # print(f"Final: {sim.get_info()}")


if __name__ == "__main__":
    main()