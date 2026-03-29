## Simulation of a lasso with a patch of gecko adhesive on the end capturing a satellite
## Developed by M. Coughlin for the BDML research group at Stanford University
## 2026
import os
from pathlib import Path

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation

class InertialBody:
    def __init__(self, model_path: Path, init_pos: np.ndarray = np.array((0, 0, 0)),
                 init_orientation: np.ndarray = np.array((0, 0, 0)),
                 init_lin_vel: np.ndarray = np.array((0, 0, 0)),
                 init_angle_vel: np.ndarray = np.array((0, 0, 0)),
                 inertia_tensor: np.ndarray = None, mass: float = None):
        # Saves the model geometry
        self.model = trimesh.load(model_path, force="mesh")

        # Defines the state of the model with the center of mass position and a quaternion
        self.pos = init_pos.astype(np.float32)
        self.rot = Rotation.from_euler("XYZ", init_orientation)

        # Defines the velocities of the body
        self.lin_vel = init_lin_vel.astype(np.float32)
        self.ang_vel = init_angle_vel.astype(np.float32)

        # Unless mass properties are specified, compute them from the mesh
        # Inertia tensor
        if inertia_tensor is None:
            self.I_ten = trimesh.inertia.points_inertia(self.model.vertices)
        else:
            self.I_ten = inertia_tensor

        self.inv_I_ten = np.linalg.inv(self.I_ten)

        # Mass
        if mass is None:
            self.mass = self.model.mass
        else:
            self.mass = mass

    def _update_pos(self, dt: float):
        self.pos += self.lin_vel * dt

        # Angular velocity is updated twice -> once based on any external forces, and once based on Euler coupling.
        # Compute and update the angular velocity based on Euler coupling
        alpha = self.inv_I_ten @ np.cross(-1 * self.ang_vel, self.I_ten @ self.ang_vel)
        self.ang_vel += alpha * dt

        # Compute and update the rotation angle
        ang_delta = self.ang_vel * dt
        self.rot *= Rotation.from_rotvec(ang_delta)



def show_sim(bodies: list[InertialBody], dt: float):
    # Create a scene
    scene = trimesh.Scene()

    # Use a dictionary to map InertialBody instances to their node names in the scene
    node_names = {}
    for i, body in enumerate(bodies):
        node_name = f"body_{i}"
        node_names[id(body)] = node_name
        scene.add_geometry(body.model, node_name=node_name)

    def update_scene(scene_to_update):
        """
        Callback function executed for each frame of the animation.
        This function updates the physics and transforms of all bodies.
        """
        for body in bodies:
            # 1. Update the body's physics (position and rotation)
            body._update_pos(dt)

            # 2. Create a 4x4 transformation matrix from the new state
            transform = np.eye(4)
            transform[:3, :3] = body.rot.as_matrix()
            transform[:3, 3] = body.pos

            # 3. Update the transform of the corresponding model in the scene
            node_name = node_names[id(body)]
            scene_to_update.graph.update(node_name, matrix=transform)

    # The `scene.show()` method with a callback will open a window and
    # repeatedly call the `update_scene` function.
    # The window will run until the user closes it.
    print("Starting simulation... Close the viewer window to exit.")
    scene.show(callback=update_scene)


if __name__ == '__main__':
    model_dir = Path(os.getcwd()) / "Assets"

    satellite = InertialBody(model_dir / "SatelliteNGPayload.obj", np.array((0, 0, 0)))
    av = InertialBody(model_dir / "AV_1_0p5_0p5.obj", np.array((-20, -20, 4)))


    # Time step for the simulation (e.g., for ~60 FPS animation)
    DT = 1.0 / 60.0

    # Run the visualization
    show_sim([satellite, av], DT)




