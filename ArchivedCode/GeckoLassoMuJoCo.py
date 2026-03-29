## Simulation of a lasso with a patch of gecko adhesive on the end capturing a satellite
## Developed by M. Coughlin for the BDML research group at Stanford University
## Built using the MuJoCo physics engine
## 2026
## MODIFIED to simulate a force-controlled winch/pulley on the AV.

import os
from pathlib import Path
from time import sleep

import mujoco.viewer
import numpy as np
import trimesh


def conv_pos_str_to_np_arr(pos_str) -> np.ndarray:
    return np.array(eval("[" + pos_str.replace(" ", ",") + "]"), dtype=np.float32)


def generate_chain_xml(num_links: int, link_length: float, link_radius: float, sat_anchor_str: str,
                       init_cable_dir: np.array) -> tuple[str, str]:
    body_xml = ""
    tendon_sites = []
    indent_base = "    "

    cable_dir_str = str((init_cable_dir * link_length).tolist()).replace("[", "").replace("]", "").replace(",", "")

    for i in range(num_links):
        body_name = f"link_{i}"
        geom_name = f"geom_link_{i}"
        site_name = f"site_link_{i}"
        tendon_sites.append(site_name)
        indent = indent_base * (i + 1)
        if i == 0:
            pos = sat_anchor_str
        else:
            pos = cable_dir_str

        body_xml += f'{indent}<body name="{body_name}" pos="{pos}">\n'
        body_xml += f'{indent}    <joint type="ball"/>\n'
        body_xml += f'{indent}    <geom name="{geom_name}" type="sphere" size="{link_radius}" contype="1" conaffinity="1"/>\n'
        body_xml += f'{indent}    <site name="{site_name}" pos="0 0 0" size="0.005"/>\n'

    for i in range(num_links):
        body_xml += f'{indent_base * (num_links - i)}</body>\n'

    total_length = num_links * link_length
    # Note: The 'range' in the tendon is now a soft constraint. The actuator will pull on it.
    tendon_xml = f'<spatial name="string_tendon" limited="true" range="0 {total_length}" width="{link_radius / 2}" rgba="0.8 0.2 0.2 1">\n'
    tendon_xml += '    <site site="sat_anchor"/>\n'
    for site_name in tendon_sites:
        tendon_xml += f'    <site site="{site_name}"/>\n'

    tendon_xml += '    <site site="av_anchor"/>\n'
    tendon_xml += '</spatial>'


    return body_xml, tendon_xml


def load_mesh(path: Path):
    # Loads a mesh from a .obj file. This function ensures that the origin of the body is placed at the center of mass.
    mesh = trimesh.load(path, force="mesh")
    mesh.apply_translation(-mesh.center_mass)
    temp_export_file = path.parent / ("._Repositioned_" + path.name)
    mesh.export(temp_export_file)
    return temp_export_file


def get_free_joint_qvel_indices(model: mujoco.MjModel, body_name: str) -> tuple[slice, slice]:
    # A free joint is always the first and only joint for a free body.
    joint_id = model.body(body_name).jntadr[0]
    qvel_start_addr = model.jnt_dofadr[joint_id]
    linear_slice = slice(qvel_start_addr, qvel_start_addr + 3)
    angular_slice = slice(qvel_start_addr + 3, qvel_start_addr + 6)
    return linear_slice, angular_slice


def get_free_joint_qpos_indices(model: mujoco.MjModel, body_name: str) -> tuple[slice, slice]:
    # A free joint is always the first and only joint for a free body.
    joint_id = model.body(body_name).jntadr[0]
    qpos_start_addr = model.jnt_qposadr[joint_id]
    linear_slice = slice(qpos_start_addr, qpos_start_addr + 3)
    angular_slice = slice(qpos_start_addr + 3, qpos_start_addr + 7)
    return linear_slice, angular_slice


def set_initial_state(model: mujoco.MjModel, data: mujoco.MjData, body_name: str,
                      lin_vel: np.ndarray = np.array((0, 0, 0), dtype=np.float32),
                      ang_vel: np.ndarray = np.array((0, 0, 0), dtype=np.float32),
                      lin_pos: np.ndarray = np.array((0, 0, 0), dtype=np.float32),
                      ang_pos: np.ndarray = np.array((0, 0, 0), dtype=np.float32)):
    # Sets the angular and linear velocity
    lin_slice, ang_slice = get_free_joint_qvel_indices(model, body_name)
    data.qvel[lin_slice] = lin_vel
    data.qvel[ang_slice] = ang_vel

    # Sets the angular and linear position
    lin_slice, ang_slice = get_free_joint_qpos_indices(model, body_name)
    data.qpos[lin_slice] = lin_pos
    mujoco.mju_euler2Quat(data.qpos[ang_slice], ang_pos, 'xyz')

    # Update the model.
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)


if __name__ == "__main__":
    ##########
    # Configuration
    ##########

    # Set path to the base directory
    work_dir = Path(os.getcwd())

    # Set path to key files, including the XML defining the simulation and the two mesh files
    sim_def_path = work_dir / "SimDefinitions" / "SpinningSatWithAV.xml"
    sat_obj_path = work_dir / "Assets" / "SatelliteNGPayload.obj"
    av_obj_path = work_dir / "Assets" / "AV_1_0p5_0p5.obj"

    # Physics / kinematics / satellite dynamics
    anchor_pt = "1.8 0 0"
    av_init_pos = np.array((7, 7, 0), dtype=np.float32)
    NUM_LINKS = 30
    LINK_LENGTH = 0.5  # Must be defined for generate_chain_xml

    ##########
    # Prepare to run the simulation
    ##########

    # Generates the XML which defines the string
    conn_uv = av_init_pos - conv_pos_str_to_np_arr(anchor_pt)
    conn_uv /= np.linalg.norm(conn_uv)
    chain_body_xml, chain_tendon_xml = generate_chain_xml(NUM_LINKS, LINK_LENGTH, 0.01, anchor_pt, conn_uv)

    # Load the XML
    sim_def = open(sim_def_path).read()
    sim_def = sim_def.format(sat_obj_file=str(load_mesh(sat_obj_path)),
                             av_obj_file=str(load_mesh(av_obj_path)),
                             sat_anchor_pos=anchor_pt,
                             chain_bodies=chain_body_xml,
                             chain_tendons=chain_tendon_xml)

    print(sim_def)

    # Load the simulation
    sim_model = mujoco.MjModel.from_xml_string(sim_def)
    sim_data = mujoco.MjData(sim_model)

    ##########
    # Sets the initial conditions
    ##########
    set_initial_state(sim_model, sim_data, "sat_body",
                      lin_vel=np.array((0, 0, 0), dtype=np.float32),
                      ang_vel=np.array((0, 0, 0), dtype=np.float32),  # Give sat some spin
                      lin_pos=np.array((0, 0, 0), dtype=np.float32))

    set_initial_state(sim_model, sim_data, "av_body",
                      lin_vel=np.array((0, 0, 0), dtype=np.float32),  # Give AV some velocity away from sat
                      ang_vel=np.array((0, 0, 0), dtype=np.float32),
                      lin_pos=av_init_pos)

    ##########
    # Runs the simulation
    ##########

    # Get slices and IDs for easy access and diagnostics
    _, ang_vel_slice = get_free_joint_qvel_indices(sim_model, "sat_body")
    sat_body_id = mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_BODY, "sat_body")

    # --- MODIFIED: Controlled Tension Parameters ---
    # The AV will try to apply this much pulling force via the winch.
    # Note: The actual tension will depend on system dynamics.
    # This is the "commanded" tension.
    CONTROLLED_TENSION_FORCE = 0  # (Newtons)

    # Get the ID of the actuator we defined in the XML.
    try:
        winch_actuator_id = mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "winch")
    except ValueError:
        print("\nERROR: Could not find the 'winch' actuator in the model.")
        print("Please ensure you have added the <actuator> block to your XML file.\n")
        exit()
    # --- END MODIFICATION ---

    with mujoco.viewer.launch_passive(sim_model, sim_data) as viewer:
        step = 0
        while viewer.is_running():
            step_start = sim_data.time
            while sim_data.time - step_start < 1.0 / 60.0:  # Simulate at 60Hz
                sleep(1)
                # --- MODIFIED: Apply Controlled Tension ---
                # The control input for a tendon actuator is the desired force.
                # A negative value pulls (reels in). We set the winch to pull with our target force.
                sim_data.ctrl[winch_actuator_id] = -CONTROLLED_TENSION_FORCE
                print(sim_data.ctrl)
                # --- STEP THE PHYSICS ENGINE ---
                # MuJoCo automatically calculates and applies the forces on all relevant
                # bodies based on the actuator's commanded force. No manual force calculation needed.
                mujoco.mj_step(sim_model, sim_data)
                viewer.sync()

            # Diagnostic printing from original code
            if step % 1 == 0:  # Print once per second
                current_ang_vel = sim_data.qvel[ang_vel_slice]
                current_ang_mom = sim_data.subtree_angmom[sat_body_id]
                ang_vel_magnitude = np.linalg.norm(current_ang_vel)
                ang_mom_magnitude = np.linalg.norm(current_ang_mom)

                print(f"--- Time: {sim_data.time:.2f}s ---")
                print(f"Angular Velocity (ω): {ang_vel_magnitude:.4f} rad/s")
                print(f"Angular Momentum (L): {ang_mom_magnitude:.4f} kg·m²/s")
                print("-" * 20)

            viewer.sync()
            step += 1