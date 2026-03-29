## Simulation of a lasso with a patch of gecko adhesive on the end capturing a satellite
## Developed by M. Coughlin for the BDML research group at Stanford University
## Built using the MuJoCo physics engine
## 2026

import os
from pathlib import Path
from time import sleep

import mujoco.viewer
import numpy as np
import trimesh


def conv_pos_str_to_np_arr(pos_str) -> np.ndarray:
    return np.array(eval("[" + pos_str.replace(" ", ",") + "]"), dtype=np.float32)


def generate_chain_xml(num_links: int, link_length: float, link_radius: float, sat_anchor_str: str, init_cable_dir: np.array) -> tuple[str, str]:
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
    tendon_xml = f'<spatial name="string_tendon" limited="true" range="0 {total_length}" width="{link_radius / 2}" rgba="0.8 0.2 0.2 1">\n'
    tendon_xml += '    <site site="sat_anchor"/>\n'
    for site_name in tendon_sites:
        tendon_xml += f'    <site site="{site_name}"/>\n'
    tendon_xml += '</spatial>'

    return body_xml, tendon_xml


def load_mesh(path: Path):
    # Loads a mesh from a .obj file. This function ensures that the origin of the body is placed at the center of mass,
    # which saves us from having to apply many corrections in MuJoCo.

    # Mass properties can also be manually specified, which would be necessary for a true satellite assembly
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
    # data.qpos[ang_slice] = ang_pos


    # quat = np.zeros(4)
    mujoco.mju_euler2Quat(data.qpos[ang_slice], ang_pos, 'xyz')

    # Update the model. This is necessary to compute the proper momentum values.
    mujoco.mj_forward(sim_model, sim_data)
    mujoco.mj_subtreeVel(sim_model, sim_data)


def solve_ik_manual(model, data, end_effector_site_id, target_pos, iterations=100, tolerance=1e-5, damping=1e-3):
    """
    Performs inverse kinematics using the Damped Least Squares (DLS) method.
    This function is a manual implementation and does not depend on mj_ik or mj_solveIK.
    """
    # Get the qpos slice for all ball joints in the chain
    num_links = 0
    qpos_indices = []
    for i in range(model.njnt):  # Iterate through all joints in the model
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name and 'link' in joint_name:
            num_links += 1
            # A ball joint has 3 DoFs, but its qpos is a 4-element quaternion
            start_adr = model.jnt_qposadr[i]
            qpos_indices.extend(range(start_adr, start_adr + 4))

    if not num_links:
        print("Error: No 'link' joints found for IK.")
        return False

    nv = model.nv  # Total number of velocity DoFs

    for i in range(iterations):
        # Calculate the error (desired change in position)
        current_pos = data.site(end_effector_site_id).xpos
        dx = target_pos - current_pos

        # Check for convergence
        if np.linalg.norm(dx) < tolerance:
            print(f"IK converged in {i + 1} iterations.")
            return True

        # --- Calculate the Jacobian ---
        # Allocate memory for the Jacobian matrices
        jac_pos = np.zeros((3, nv))
        jac_rot = np.zeros((3, nv))

        # This function populates jac_pos and jac_rot
        mujoco.mj_jac(model, data, jac_pos, jac_rot, current_pos, end_effector_site_id)

        # We only care about the positional Jacobian
        J = jac_pos

        # --- Solve using Damped Least Squares (DLS) ---
        J_T = J.T
        lambda_sq = damping ** 2

        # Solve (J * J' + lambda^2 * I) * x = dx
        # This is more stable than calculating the inverse directly
        A = J @ J_T + lambda_sq * np.identity(3)
        b = dx

        # Solve for the intermediate vector 'y' where y = (JJ' + l^2I)^-1 * dx
        y = np.linalg.solve(A, b)

        # dq = J' * y
        dq = J_T @ y

        # --- Apply the solution ---
        # mj_integratePos expects a full 'nv' dimensional velocity vector
        # and applies it to the 'nq' dimensional qpos vector.
        mujoco.mj_integratePos(model, data.qpos, dq, 1.0)

        # Update the simulation state with the new qpos
        mujoco.mj_forward(model, data)

    print("Warning: IK did not converge after max iterations.")
    return False



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

    ##########
    # Prepare to run the simulation
    ##########

    # Generates the XML which defines the string
    conn_uv = av_init_pos - conv_pos_str_to_np_arr(anchor_pt)
    conn_uv /= np.linalg.norm(conn_uv)
    chain_body_xml, chain_tendon_xml = generate_chain_xml(NUM_LINKS, 0.5, 0.01, anchor_pt, conn_uv)

    # Load the XML
    sim_def = open(sim_def_path).read()
    sim_def = sim_def.format(sat_obj_file=str(load_mesh(sat_obj_path)),
                             av_obj_file=str(load_mesh(av_obj_path)),
                             sat_anchor_pos=anchor_pt,
                             chain_bodies=chain_body_xml,
                             chain_tendons=chain_tendon_xml)

    # Load the simulation
    sim_model = mujoco.MjModel.from_xml_string(sim_def)
    sim_data = mujoco.MjData(sim_model)

    ##########
    # Sets the initial conditions
    ##########
    # set_body_angular_velocity(sim_model, sim_data, "av_body", np.array((0, 0, 0), dtype=np.float32))
    # increment_body_linear_velocity(sim_model, sim_data, "av_body", np.array((0, 0, 0), dtype=np.float32))


    set_initial_state(sim_model, sim_data, "sat_body",
                      np.array((0, 0, 0), dtype=np.float32),
                      # np.array((0, 0.01, 3), dtype=np.float32),
                      np.array((0, 0, 0), dtype=np.float32),
                      np.array((0, 0, 0), dtype=np.float32),
                      np.array((0, 0, 0), dtype=np.float32))


    set_initial_state(sim_model, sim_data, "av_body",
                      np.array((0, 0, 0), dtype=np.float32),
                      np.array((0, 0, 0), dtype=np.float32),
                      av_init_pos,
                      np.array((0, 0, 0), dtype=np.float32))

    ##########
    # Runs the simulation
    ##########

    # Get the qvel slice for easy access to angular velocity
    _, ang_vel_slice = get_free_joint_qvel_indices(sim_model, "sat_body")
    sat_body_id = 1

    SPRING_STIFFNESS = 500.0  # (N/m) How strong the pull is.
    DAMPING_COEFFICIENT = 20.0   # (N-s/m) Resists oscillation, adds stability.

    av_body_id = mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_BODY, "av_body")
    link_body_ids = [mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_BODY, f"link_{i}") for i in range(NUM_LINKS)]


    with mujoco.viewer.launch_passive(sim_model, sim_data) as viewer:
        step = 0
        while viewer.is_running():
            print(step)
            step_start = sim_data.time
            while sim_data.time - step_start < 1.0 / 60.0:  # Simulate at 60Hz
                """
                # --- ZERO OUT EXTERNAL FORCES AT THE START OF THE STEP ---
                # This is crucial to prevent forces from accumulating over time.
                sim_data.xfrc_applied[:] = 0

                # --- FIND THE DEPARTURE POINT ---
                av_pos = sim_data.body(av_body_id).xpos
                min_dist_sq = float('inf')
                departure_link_id = -1

                for i in range(NUM_LINKS):
                    link_pos = sim_data.body(link_body_ids[i]).xpos
                    dist_sq = np.sum((av_pos - link_pos) ** 2)
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        departure_link_id = link_body_ids[i]

                min_dist = np.sqrt(min_dist_sq)
                departure_point_pos = sim_data.body(departure_link_id).xpos

                # --- CALCULATE AND APPLY SPRING-DAMPER FORCE ---
                if departure_link_id != -1:
                    # 1. Calculate Spring Force (Hooke's Law)
                    force_direction = (departure_point_pos - av_pos) / (min_dist + 1e-9)
                    spring_force_magnitude = SPRING_STIFFNESS * min_dist

                    # 2. Calculate Damping Force
                    av_vel = sim_data.body(av_body_id).cvel[3:]  # Linear velocity part
                    link_vel = sim_data.body(departure_link_id).cvel[3:]
                    relative_vel = np.dot(link_vel - av_vel, force_direction)
                    damping_force_magnitude = DAMPING_COEFFICIENT * relative_vel

                    # 3. Total force
                    total_force_magnitude = spring_force_magnitude + damping_force_magnitude
                    force_vector = force_direction * total_force_magnitude

                    # 4. Apply forces
                    # Apply force to the AV
                    sim_data.xfrc_applied[av_body_id, 0:3] = force_vector
                    # Apply equal and opposite force to the departure link
                    sim_data.xfrc_applied[departure_link_id, 0:3] = -force_vector

                # --- STEP THE PHYSICS ENGINE ---
                """
                mujoco.mj_step(sim_model, sim_data)
                print(sim_data.time)

            if step % 10 == 0:  # Print every 10 steps to avoid spamming
                # 1. Get the current angular velocity (ω)
                current_ang_vel = sim_data.qvel[ang_vel_slice]

                # 2. Get the current angular momentum (L)
                current_ang_mom = sim_data.subtree_angmom[sat_body_id]

                # 3. Calculate the magnitudes
                ang_vel_magnitude = np.linalg.norm(current_ang_vel)
                ang_mom_magnitude = np.linalg.norm(current_ang_mom)

                print(f"--- Step {step} ---")
                print(f"Angular Velocity (ω): {current_ang_vel}, Magnitude: {ang_vel_magnitude:.4f}")
                print(f"Angular Momentum (L): {current_ang_mom}, Magnitude: {ang_mom_magnitude:.4f}")
                print("-" * 20)

            viewer.sync()
            step += 1
    pass
