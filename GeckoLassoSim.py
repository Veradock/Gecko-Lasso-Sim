"""
Gecko Lasso Simulation — simulation of a cable wrapping, capturing, and detumbling a satellite
M. Coughlin 2026 created for the BDML research group at Stanford University

This code is designed to use the MuJoCo native physics engine as much as possible

Requirements:
    pip install mujoco numpy matplotlib trimesh

Run:
    python GeckoLassoSim.py
"""

# TODO: clean up the code, clean up the visualization, and record!!

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
    return -10, 0, 0

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
        toward_origin = self._get_av_pos() - spool_pos
        dist = np.linalg.norm(toward_origin)

        # if dist > 1e-6:
        direction = toward_origin / dist
        force = direction * SPOOL_TENSION

        # Clear any existing applied forces first
        self.data.xfrc_applied[body_id, :] = 0

        # Use mj_applyFT to properly apply force and torque
        torque_vec = np.zeros(3, dtype=np.float64)

        # Create array to receive the generalized forces
        # Applies the forces to the rope
        qfrc_applied = np.zeros(self.model.nv, dtype=np.float64)
        mujoco.mj_applyFT(self.model, self.data, force, torque_vec, spool_pos, body_id, qfrc_applied)

        # Now apply the reaction forces on the AV
        mujoco.mj_applyFT(self.model, self.data, -1 * force + get_thruster_reaction(), torque_vec,
                          self._get_av_pos(), self.av_body_id, qfrc_applied)
        self.data.qfrc_applied[:] = qfrc_applied  # Note this adds instead of re

    def _get_av_pos(self):
        start_addr = self.model.jnt_dofadr[self.av_joint_id]
        return self.data.qpos[start_addr:start_addr+3]

    def _maybe_spawn(self):
        if self.active_count < MAX_SEGMENTS:
            spool_pos = self._get_pos(self.spool_idx)

            # Calculate free link length (spool to origin)
            free_link_length = np.linalg.norm(self._get_av_pos() - spool_pos)

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
        spawn_pos = old_spool_pos + toward_origin_dir * EQUILIBRIUM_LINK_LENGTH

        # Set new segment position and orientation
        self.data.qpos[qpos_adr:qpos_adr+3] = spawn_pos
        self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

        # To result in smoother spawning, we will give the cable an initial velocity
        self.data.qvel[qvel_adr:qvel_adr+3] = self.data.qvel[old_qvel_adr:old_qvel_adr+3]

        # Activate tendon between old spool and new segment
        tendon_id = self.cable_tendon_ids[old_spool]
        self.model.tendon_stiffness[tendon_id] = SPRING_STIFFNESS
        self.model.tendon_damping[tendon_id] = SPRING_DAMPING

        # Update visual colors
        self.model.geom_rgba[self.cable_geom_ids[new_idx]] = [0, 1, 0, 1]
        self.model.geom_rgba[self.cable_geom_ids[old_spool]] = [1, 1, 0, 1]

        # Update spool index and active count
        self.spool_idx = new_idx
        self.active_count = new_idx + 1

        # Now that the new spool is spawned, forward the state again. Forwarding twice is required to reduce tension
        # spikes
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
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, CABLE_RADIUS, self._get_av_pos(), spool_pos)
        g.rgba[:] = [0.7, 0.1, 0.1, 1.0]
        scene.ngeom += 1

        for i in range(self.spool_idx):
            p1 = self._get_pos(i)
            p2 = self._get_pos(i + 1)

            g = scene.geoms[scene.ngeom]
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, CABLE_RADIUS, p1, p2)
            g.rgba[:] = [0.9, 0.2, 0.1, 1.0]
            scene.ngeom += 1



def main():
    print("=" * 50)
    print("Gecko Lasso Simulation with MuJoCo")
    print("=" * 50)

    sim = Simulation()

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 25
        viewer.cam.lookat[:] = [0, 0, 0]

        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        while viewer.is_running():
            sim.step()

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                sim.render_cable(viewer.user_scn)

            if sim.data.time > 1 and not hasattr(sim, 'plotted'):
                sim.plot_tension()
                sim.plotted = True

            viewer.sync()



if __name__ == "__main__":
    main()