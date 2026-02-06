"""
MuJoCo Cable Wrapping Simulation - Native Physics Version

A cable unspools from a fixed point and wraps around a spinning cylinder.
Uses MuJoCo's native contact and friction calculations.

Requirements:
    pip install mujoco numpy matplotlib

Run:
    python cable_wrapping_sim.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

CYLINDER_RADIUS = 2.0      # meters
CYLINDER_HEIGHT = 10.0     # meters
CYLINDER_OMEGA = 1.0       # rad/s rotation speed

CABLE_ORIGIN = np.array([10.0, 0.0, 0.0])  # Fixed point in space
ATTACHMENT_HEIGHT = -2.0   # 2m below origin (origin at z=0, attachment at z=-2)

CABLE_SEGMENT_LENGTH = 0.3  # Length of each cable segment
CABLE_RADIUS = 0.03         # Physical radius of cable
NODE_RADIUS = 0.04          # Visual radius of nodes

MAX_SEGMENTS = 50           # Maximum number of pre-allocated cable segments

# Cable physical properties
CABLE_MASS = 0.05           # Mass per segment (kg)

# Friction coefficients (MuJoCo format: sliding, torsional, rolling)
CABLE_FRICTION = [0.5, 0.01, 0.01]


def create_model_xml():
    # Initial attachment point position
    attach_pos = np.array([CYLINDER_RADIUS, 0.0, ATTACHMENT_HEIGHT])

    # Vector from attachment toward origin
    to_origin_vec = CABLE_ORIGIN - attach_pos
    to_origin_dir = to_origin_vec / np.linalg.norm(to_origin_vec)

    # Direction behind the spool-out point (away from cylinder, past origin)
    behind_dir = to_origin_dir

    # Number of segments: calculate so chain ends before origin with room for free link
    # Distance from attach to origin
    attach_to_origin = np.linalg.norm(CABLE_ORIGIN - attach_pos)

    # For equilibrium with 10N tension and k=10000 N/m:
    SPOOL_TENSION = 10.0
    SPRING_STIFFNESS = 10000.0
    EQUILIBRIUM_STRETCH_PER_LINK = SPOOL_TENSION / SPRING_STIFFNESS  # 0.001m
    EQUILIBRIUM_LINK_LENGTH = CABLE_SEGMENT_LENGTH + EQUILIBRIUM_STRETCH_PER_LINK  # 0.301m

    # Calculate number of segments so spool is ~0.5m before origin (leaving room for free link)
    desired_chain_length = attach_to_origin - 0.5  # leave 0.5m for free link
    init_segments = int(desired_chain_length / EQUILIBRIUM_LINK_LENGTH) + 1  # +1 because we count nodes not links

    # Calculate positions for each node at equilibrium spacing
    active_positions = []
    for i in range(init_segments):
        pos = attach_pos + to_origin_dir * (i * EQUILIBRIUM_LINK_LENGTH)
        active_positions.append(pos)

    active_positions[0] = attach_pos.copy()

    # Spool is at the end of the chain - free link goes from spool to origin
    spool_pos = active_positions[-1]
    free_link_length = np.linalg.norm(CABLE_ORIGIN - spool_pos)

    print(f"DEBUG: attach_to_origin={attach_to_origin:.4f}m")
    print(f"DEBUG: init_segments={init_segments}")
    print(f"DEBUG: equilibrium link length={EQUILIBRIUM_LINK_LENGTH:.4f}m")
    print(f"DEBUG: spool pos={spool_pos}")
    print(f"DEBUG: free link length={free_link_length:.4f}m")

    # Build cable bodies with sites
    cable_bodies = ""
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
            offset = (i - init_segments + 1) * CABLE_SEGMENT_LENGTH
            pos = CABLE_ORIGIN + behind_dir * offset
            color = "0.5 0.5 0.5 1"

        cable_bodies += f"""
        <body name="cable_{i}" pos="{pos[0]} {pos[1]} {pos[2]}">
            <freejoint name="cable_free_{i}"/>
            <geom name="cable_geom_{i}" type="sphere" size="{NODE_RADIUS}" mass="{CABLE_MASS}"
                  friction="{CABLE_FRICTION[0]} {CABLE_FRICTION[1]} {CABLE_FRICTION[2]}"
                  rgba="{color}" contype="1" conaffinity="2"/>
            <site name="cable_site_{i}" pos="0 0 0" size="0.01"/>
        </body>
"""

    # Build spatial tendons connecting adjacent segments
    # Only initially active tendons are between active segments
    SPRING_STIFFNESS = 10000.0  # N/m - stiff spring
    SPRING_DAMPING = 1000.0     # Ns/m - increased damping for stability

    cable_tendons = ""
    for i in range(MAX_SEGMENTS - 1):
        # Use range attribute to limit tendon force to tension only (no compression)
        # Set springlength to create rest length of CABLE_SEGMENT_LENGTH
        cable_tendons += f"""
        <spatial name="cable_tendon_{i}" springlength="{CABLE_SEGMENT_LENGTH}" stiffness="{SPRING_STIFFNESS}" damping="{SPRING_DAMPING}" limited="true" range="0 100">
            <site site="cable_site_{i}"/>
            <site site="cable_site_{i+1}"/>
        </spatial>"""

    # Attachment constraint only - connect segment 0 to cylinder
    xml = f"""
<mujoco model="cable_wrapping_springs">
    <option gravity="0 0 0" timestep="0.002" integrator="implicit" cone="elliptic" impratio="10">
        <flag warmstart="enable"/>
    </option>
    
    <size nconmax="500" njmax="2000" nstack="2000000"/>
    
    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
        <map force="0.1" zfar="100"/>
    </visual>
    
    <default>
        <geom condim="4" friction="{CABLE_FRICTION[0]} {CABLE_FRICTION[1]} {CABLE_FRICTION[2]}"/>
    </default>
    
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.1 0.1 0.2" rgb2="0.3 0.3 0.4" width="512" height="512"/>
        <material name="cylinder_mat" rgba="0.3 0.5 0.8 0.8"/>
        <material name="ground_mat" rgba="0.2 0.2 0.2 1" reflectance="0.1"/>
    </asset>
    
    <worldbody>
        <geom type="plane" size="20 20 0.1" pos="0 0 -6" material="ground_mat" contype="0" conaffinity="0"/>
        
        <site name="cable_origin" pos="{CABLE_ORIGIN[0]} {CABLE_ORIGIN[1]} {CABLE_ORIGIN[2]}" size="0.06" rgba="0 1 0 1"/>
        
        <body name="cylinder" pos="0 0 0">
            <joint name="cylinder_rotation" type="hinge" axis="0 0 1" damping="0"/>
            <geom name="cylinder_geom" type="cylinder" size="{CYLINDER_RADIUS} {CYLINDER_HEIGHT/2}" material="cylinder_mat" 
                  contype="2" conaffinity="1" friction="{CABLE_FRICTION[0]} {CABLE_FRICTION[1]} {CABLE_FRICTION[2]}" condim="4"/>
            <site name="attachment_point" pos="{CYLINDER_RADIUS} 0 {ATTACHMENT_HEIGHT}" size="0.05" rgba="1 0.5 0 1"/>
            <body name="attachment_body" pos="{CYLINDER_RADIUS} 0 {ATTACHMENT_HEIGHT}">
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
    
    <actuator>
        <velocity name="cylinder_motor" joint="cylinder_rotation" kv="10000"/>
    </actuator>
</mujoco>
"""
    return xml, init_segments


class Simulation:
    def __init__(self):
        self.xml, self.init_segments = create_model_xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)

        # Forward pass to initialize physics state consistently
        mujoco.mj_forward(self.model, self.data)

        self.cylinder_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_rotation")
        self.motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cylinder_motor")

        self.cable_body_ids = []
        self.cable_geom_ids = []
        for i in range(MAX_SEGMENTS):
            self.cable_body_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"cable_{i}"))
            self.cable_geom_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"cable_geom_{i}"))

        # Get tendon IDs for spring connections
        self.cable_tendon_ids = []
        for i in range(MAX_SEGMENTS - 1):
            self.cable_tendon_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, f"cable_tendon_{i}"))

        # Deactivate tendons for inactive segments by setting stiffness to 0
        for i in range(self.init_segments - 1, MAX_SEGMENTS - 1):
            tendon_id = self.cable_tendon_ids[i]
            self.model.tendon_stiffness[tendon_id] = 0.0
            self.model.tendon_damping[tendon_id] = 0.0

        self.spool_idx = self.init_segments - 1
        self.active_count = self.init_segments
        self.time = 0.0
        self.wrap_count = 0.0

        # Debug tracking for first two spawns
        self.debug_first_spawn_time = None
        self.debug_first_spawn_new_idx = None
        self.debug_second_spawn_time = None
        self.debug_second_spawn_new_idx = None

        # Tension tracking
        self.tension_history = []
        self.time_history = []

        print(f"Initialized: {self.active_count} segments, spool={self.spool_idx}")

    def _get_pos(self, idx):
        return self.data.xpos[self.cable_body_ids[idx]].copy()

    def step(self):
        # TEMPORARILY DISABLE CYLINDER ROTATION for debugging
        self.data.ctrl[self.motor_id] = CYLINDER_OMEGA  # Was: CYLINDER_OMEGA

        # Apply tension force on spool - must be done before mj_step
        self._apply_spool_tension()

        # Debug: investigate force balance on spool
        if self.time < 0.02 or (self.time < 0.2 and int(self.time * 100) % 5 == 0):
            self._debug_force_balance()

        mujoco.mj_step(self.model, self.data)

        self._maybe_spawn()
        self._debug_first_spawn()
        self._record_tension()
        self.wrap_count = self.data.qpos[self.cylinder_jnt_id] / (2 * np.pi)
        self.time = self.data.time

    def _record_tension(self):
        if self.time > 18.0:
            return

        self.time_history.append(self.time)

        # Debug: print detailed info at specific times
        debug_times = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]
        should_debug = any(abs(self.time - t) < 0.003 for t in debug_times)

        if should_debug:
            print(f"\n=== TENSION DEBUG t={self.time:.3f}s ===")
            print(f"active_count={self.active_count}, spool_idx={self.spool_idx}")

            # Check spool position
            spool_pos = self._get_pos(self.spool_idx)
            print(f"spool pos: {spool_pos}")
            print(f"distance to origin: {np.linalg.norm(CABLE_ORIGIN - spool_pos):.4f}m")

            # Check last few tendon tensions
            print("Tendon tensions (last 5 active):")
            for i in range(max(0, self.active_count - 6), self.active_count - 1):
                tendon_id = self.cable_tendon_ids[i]
                length = self.data.ten_length[tendon_id]
                stiffness = self.model.tendon_stiffness[tendon_id]
                stretch = length - CABLE_SEGMENT_LENGTH
                force = stiffness * stretch
                print(f"  tendon {i}: length={length:.4f}, stretch={stretch:.6f}, force={force:.2f}N")

        # Get tendon forces
        tensions = []
        for i in range(self.active_count - 1):
            if i < len(self.cable_tendon_ids):
                tendon_id = self.cable_tendon_ids[i]
                length = self.data.ten_length[tendon_id]
                stiffness = self.model.tendon_stiffness[tendon_id]
                stretch = length - CABLE_SEGMENT_LENGTH
                force = stiffness * stretch
                tensions.append(force)
            else:
                tensions.append(0.0)

        # Add free link tension - the 10N applied force
        tensions.append(10.0)  # We always apply 10N

        self.tension_history.append(tensions)

    def _apply_spool_tension(self):
        SPOOL_TENSION = 10.0  # Newtons

        spool_pos = self._get_pos(self.spool_idx)
        body_id = self.cable_body_ids[self.spool_idx]

        # Direction from spool TOWARD origin (along the free link)
        # This pulls the spool away from the chain/cylinder, creating tension
        toward_origin = CABLE_ORIGIN - spool_pos
        dist = np.linalg.norm(toward_origin)

        if dist > 1e-6:
            direction = toward_origin / dist
            force = direction * SPOOL_TENSION

            # Clear any existing applied forces first
            self.data.xfrc_applied[body_id, :] = 0

            # Use mj_applyFT to properly apply force and torque
            force_vec = np.array(force, dtype=np.float64)
            torque_vec = np.zeros(3, dtype=np.float64)
            point = np.array(spool_pos, dtype=np.float64)

            # Create array to receive the generalized forces
            qfrc_applied = np.zeros(self.model.nv, dtype=np.float64)


            mujoco.mj_applyFT(
                self.model,
                self.data,
                force_vec,
                torque_vec,
                point,
                body_id,
                qfrc_applied
            )


            # Add the computed generalized forces to data.qfrc_applied
            self.data.qfrc_applied[:] = qfrc_applied
            # mujoco.mj_forward(self.model, self.data)

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

    def _maybe_spawn(self):
        # TEMPORARILY DISABLED - chain length stays fixed
        return

    def _spawn_segment(self):
        new_idx = self.active_count
        old_spool = self.spool_idx
        spawn_number = new_idx - self.init_segments + 1

        old_spool_body_id = self.cable_body_ids[old_spool]
        old_spool_pos = self._get_pos(old_spool)
        old_spool_vel = self.data.cvel[old_spool_body_id, 3:6].copy()

        prev_seg_pos = self._get_pos(old_spool - 1)
        prev_seg_vel = self.data.cvel[self.cable_body_ids[old_spool - 1], 3:6].copy()

        print(f"SPAWN #{spawn_number}: new_idx={new_idx}, old_spool={old_spool}")
        print(f"  old_spool pos: {old_spool_pos}")
        print(f"  old_spool vel: {old_spool_vel}")
        print(f"  prev_seg ({old_spool-1}) pos: {prev_seg_pos}")
        print(f"  prev_seg ({old_spool-1}) vel: {prev_seg_vel}")
        print(f"  dist old_spool to prev_seg: {np.linalg.norm(old_spool_pos - prev_seg_pos):.4f}")

        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cable_free_{new_idx}")
        qpos_adr = self.model.jnt_qposadr[joint_id]
        qvel_adr = self.model.jnt_dofadr[joint_id]

        pre_spawn_pos = self.data.qpos[qpos_adr:qpos_adr+3].copy()
        print(f"  new segment pre-spawn pos: {pre_spawn_pos}")

        attach_pos = np.array([CYLINDER_RADIUS, 0.0, ATTACHMENT_HEIGHT])
        to_origin_dir = CABLE_ORIGIN - attach_pos
        to_origin_dir = to_origin_dir / np.linalg.norm(to_origin_dir)

        SPAWN_OFFSET = 0.15
        spawn_pos = CABLE_ORIGIN - to_origin_dir * SPAWN_OFFSET

        self.data.qpos[qpos_adr:qpos_adr+3] = spawn_pos
        self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
        self.data.qvel[qvel_adr:qvel_adr+3] = old_spool_vel
        self.data.qvel[qvel_adr+3:qvel_adr+6] = 0

        self.model.geom_rgba[self.cable_geom_ids[new_idx]] = [0, 1, 0, 1]
        self.model.geom_rgba[self.cable_geom_ids[old_spool]] = [1, 1, 0, 1]

        self.spool_idx = new_idx
        self.active_count = new_idx + 1

        if spawn_number == 1:
            self.debug_first_spawn_time = self.data.time
            self.debug_first_spawn_new_idx = new_idx
        elif spawn_number == 2:
            self.debug_second_spawn_time = self.data.time
            self.debug_second_spawn_new_idx = new_idx

    def get_info(self):
        return f"Time: {self.time:.2f}s | Wraps: {self.wrap_count:.2f} | Segments: {self.active_count}"

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
        num_constraints = len(final_tensions) - 1

        num_to_plot = min(5, num_constraints)
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
        if scene.ngeom >= scene.maxgeom:
            return

        spool_pos = self._get_pos(self.spool_idx)
        if np.all(np.isfinite(spool_pos)):
            dist = np.linalg.norm(spool_pos - CABLE_ORIGIN)
            if dist > 1e-6:
                g = scene.geoms[scene.ngeom]
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, CABLE_RADIUS,
                    CABLE_ORIGIN.reshape(3,1), spool_pos.reshape(3,1))
                g.rgba[:] = [0.7, 0.1, 0.1, 1.0]
                scene.ngeom += 1

        for i in range(self.spool_idx):
            if scene.ngeom >= scene.maxgeom:
                break

            p1 = self._get_pos(i)
            p2 = self._get_pos(i + 1)

            if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
                continue

            dist = np.linalg.norm(p2 - p1)
            if dist < 1e-6:
                continue

            g = scene.geoms[scene.ngeom]
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, CABLE_RADIUS,
                p1.reshape(3,1), p2.reshape(3,1))
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
        while viewer.is_running():
            sim.step()

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                sim.render_cable(viewer.user_scn)

            frame += 1
            if frame % 500 == 0:
                print(sim.get_info())

            if sim.time > 18 and not hasattr(sim, 'plotted'):
                sim.plot_tension()
                sim.plotted = True

            viewer.sync()
            time.sleep(0.002)

    print(f"Final: {sim.get_info()}")


if __name__ == "__main__":
    main()