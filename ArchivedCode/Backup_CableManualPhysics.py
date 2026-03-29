"""
MuJoCo Cable Wrapping Simulation

A cable unspools from a fixed point and wraps around a spinning cylinder.
The cable is discretized into segments that are dynamically spawned as needed.

Requirements:
    pip install mujoco numpy

Run:
    python cable_wrapping_sim.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

CYLINDER_RADIUS = 2.0  # meters
CYLINDER_HEIGHT = 10.0  # meters
CYLINDER_OMEGA = 1.0  # rad/s rotation speed

CABLE_ORIGIN = np.array([10.0, 0.0, 0.0])  # Fixed point in space
ATTACHMENT_HEIGHT = -2.0  # 2m below origin (origin at z=0, attachment at z=-2)

CABLE_SEGMENT_LENGTH = 0.3  # Length of each cable segment
CABLE_RADIUS = 0.03  # Visual radius of cable
CABLE_STIFFNESS = 5000.0  # Spring stiffness (reduced for stability)
CABLE_DAMPING = 500.0  # Damping coefficient

MAX_SEGMENTS = 200  # Maximum number of cable segments
SPAWN_THRESHOLD = 0.5  # Spawn new segment when stretched beyond this factor

# Substeps for cable physics (for stability with stiff springs)
CABLE_SUBSTEPS = 10

# Fixed tension in the pay-out segment (origin to first free node)
PAYOUT_TENSION = 1.0  # Newtons

# Friction coefficients
FRICTION_STATIC = 0.5  # Static friction coefficient
FRICTION_DYNAMIC = 0.3  # Dynamic friction coefficient
FRICTION_VELOCITY_THRESHOLD = 0.01  # Velocity below this is considered static


def create_base_xml():
    """Create the base MuJoCo XML with the spinning cylinder."""
    # Attachment point should start facing the cable origin (positive X direction)
    # Since origin is at (10, 0, 0), the attachment starts at (CYLINDER_RADIUS, 0, ATTACHMENT_HEIGHT)
    xml = f"""
<mujoco model="cable_wrapping">
    <option gravity="0 0 0" timestep="0.002" integrator="implicit">
        <flag warmstart="enable"/>
    </option>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
        <map force="0.1" zfar="100"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.1 0.1 0.2" rgb2="0.3 0.3 0.4" width="512" height="512"/>
        <material name="cylinder_mat" rgba="0.3 0.5 0.8 1"/>
        <material name="ground_mat" rgba="0.2 0.2 0.2 1" reflectance="0.1"/>
    </asset>

    <worldbody>
        <!-- Reference grid/ground plane for visualization -->
        <geom type="plane" size="20 20 0.1" pos="0 0 -6" material="ground_mat" contype="0" conaffinity="0"/>

        <!-- Spinning cylinder - kinematic (prescribed motion) -->
        <body name="cylinder" pos="0 0 0">
            <joint name="cylinder_rotation" type="hinge" axis="0 0 1" damping="0"/>
            <geom type="cylinder" size="{CYLINDER_RADIUS} {CYLINDER_HEIGHT / 2}" material="cylinder_mat" 
                  contype="1" conaffinity="1" friction="1 0.5 0.5" condim="4"/>

            <!-- Attachment point on cylinder surface - starts facing positive X (toward origin) -->
            <site name="attachment_point" pos="{CYLINDER_RADIUS} 0 {ATTACHMENT_HEIGHT}" size="0.12" rgba="1 0.5 0 1"/>
        </body>
    </worldbody>

    <actuator>
        <velocity name="cylinder_motor" joint="cylinder_rotation" kv="10000" forcelimited="false"/>
    </actuator>

    <sensor>
        <jointpos name="cylinder_angle" joint="cylinder_rotation"/>
        <jointvel name="cylinder_vel" joint="cylinder_rotation"/>
    </sensor>
</mujoco>
"""
    return xml


class CableNode:
    """Represents a single node in the cable."""

    def __init__(self, idx, pos):
        self.idx = idx
        self.pos = np.array(pos, dtype=np.float64)
        self.prev_pos = self.pos.copy()  # For Verlet integration
        self.fixed = False
        self.in_contact = False  # Whether node is in contact with cylinder


class CableSystem:
    """
    Manages the discretized cable system using Position-Based Dynamics (PBD).

    Uses Verlet integration with distance constraints for stability.
    This approach is much more stable than spring-mass for stiff cables.
    """

    def __init__(self, origin, initial_attachment, segment_length):
        self.origin = np.array(origin, dtype=np.float64)
        self.segment_length = segment_length

        self.nodes = []
        self.total_length = 0.0

        # Initialize cable as straight line from origin to initial attachment point
        self._initialize_cable(initial_attachment)

    def _initialize_cable(self, attachment_pos):
        """Create initial cable configuration - straight line from origin to attachment point."""
        attach_pos = np.array(attachment_pos, dtype=np.float64)

        # Create straight line from origin to attachment point
        direction = attach_pos - self.origin
        distance = np.linalg.norm(direction)

        n_segments = max(2, int(np.ceil(distance / self.segment_length)))

        self.nodes = []
        for i in range(n_segments + 1):
            t = i / n_segments
            pos = self.origin + t * direction
            node = CableNode(i, pos)
            self.nodes.append(node)

        # First node is fixed at origin
        self.nodes[0].fixed = True

        self.total_length = distance
        print(f"Initialized cable with {len(self.nodes)} nodes, length={distance:.2f}m")
        print(f"Origin: {self.origin}")
        print(f"Attachment point: {attach_pos}")

    def update(self, dt, cylinder_angle, attachment_pos):
        """
        Update cable using Position-Based Dynamics (PBD).
        Much more stable than spring-mass for stiff constraints.
        """
        if len(self.nodes) < 2:
            return

        sub_dt = dt / CABLE_SUBSTEPS

        for _ in range(CABLE_SUBSTEPS):
            # 1. Update attachment point (last node tracks cylinder anchor)
            # This is critical - the last node must follow the rotating attachment point
            self.nodes[-1].pos[:] = attachment_pos
            self.nodes[-1].prev_pos[:] = attachment_pos

            # 2. Apply fixed tension force to first free node (pay-out segment)
            if len(self.nodes) > 1:
                first_free = self.nodes[1]
                direction_to_origin = self.origin - first_free.pos
                dist = np.linalg.norm(direction_to_origin)
                if dist > 1e-8:
                    # Apply constant tension force toward origin
                    force_dir = direction_to_origin / dist
                    # F = ma, so acceleration = F/m
                    # Using Verlet: we adjust position based on acceleration
                    mass = 0.1  # mass per node
                    accel = (PAYOUT_TENSION / mass) * force_dir
                    # In Verlet, position change from acceleration is 0.5 * a * dt^2
                    first_free.pos += 0.5 * accel * sub_dt * sub_dt

            # 3. Verlet integration for free nodes (except first free node, handled above)
            damping = 0.99
            for i in range(2, len(self.nodes) - 1):
                node = self.nodes[i]
                if node.fixed:
                    continue

                vel = (node.pos - node.prev_pos) * damping
                node.prev_pos = node.pos.copy()
                node.pos = node.pos + vel

            # Also do Verlet for first free node
            if len(self.nodes) > 2:
                node = self.nodes[1]
                vel = (node.pos - node.prev_pos) * damping
                node.prev_pos = node.pos.copy()
                node.pos = node.pos + vel

            # 4. Satisfy distance constraints (multiple iterations for convergence)
            for _ in range(5):
                self._solve_constraints()

            # 5. Cylinder collision and friction
            self._handle_cylinder_collision(sub_dt)

            # 6. Keep first node at origin
            self.nodes[0].pos = self.origin.copy()
            self.nodes[0].prev_pos = self.origin.copy()

        # Check if we need to spawn new segments
        self._maybe_spawn_segment()

    def _solve_constraints(self):
        """Solve distance constraints between consecutive nodes.

        The first segment (origin to first free node) is allowed to stretch freely
        to simulate cable paying out smoothly.
        """
        # Start from segment 1 (skip segment 0 which is the pay-out segment)
        for i in range(1, len(self.nodes) - 1):
            n1 = self.nodes[i]
            n2 = self.nodes[i + 1]

            delta = n2.pos - n1.pos
            dist = np.linalg.norm(delta)

            if dist < 1e-8:
                continue

            # How much to correct
            diff = (dist - self.segment_length) / dist
            correction = delta * diff * 0.5

            # Apply correction (respecting fixed nodes)
            if not n1.fixed and not (i == len(self.nodes) - 2):
                n1.pos += correction
            if not n2.fixed and i < len(self.nodes) - 2:
                n2.pos -= correction

            # Special handling: if n2 is the attachment, only move n1
            if i == len(self.nodes) - 2:
                if not n1.fixed:
                    n1.pos += correction * 2

    def _handle_cylinder_collision(self, sub_dt):
        """Handle collision and friction between cable nodes and cylinder."""
        collision_radius = CYLINDER_RADIUS * 1.03

        for i in range(1, len(self.nodes) - 1):
            node = self.nodes[i]

            # Check if within cylinder height
            if abs(node.pos[2]) > CYLINDER_HEIGHT / 2:
                node.in_contact = False
                continue

            # Check if inside cylinder (in XY plane)
            dist_xy = np.sqrt(node.pos[0] ** 2 + node.pos[1] ** 2)

            if dist_xy < collision_radius and dist_xy > 1e-6:
                node.in_contact = True

                # Calculate normal direction (pointing outward from cylinder axis)
                normal = np.array([node.pos[0], node.pos[1], 0.0])
                normal = normal / (np.linalg.norm(normal) + 1e-8)

                # Project to surface
                node.pos[0] = normal[0] * collision_radius
                node.pos[1] = normal[1] * collision_radius

                # Calculate velocity from Verlet
                velocity = (node.pos - node.prev_pos) / sub_dt

                # Decompose velocity into normal and tangential components
                vel_normal = np.dot(velocity, normal) * normal
                vel_tangent = velocity - vel_normal
                tangent_speed = np.linalg.norm(vel_tangent)

                # Calculate normal force from tension in adjacent segments
                # Normal force comes from the cable "pressing" against the cylinder
                normal_force = self._calculate_normal_force(i, normal)

                if normal_force > 0:
                    # Calculate friction force magnitude
                    if tangent_speed < FRICTION_VELOCITY_THRESHOLD:
                        # Static friction
                        friction_coeff = FRICTION_STATIC
                    else:
                        # Dynamic friction
                        friction_coeff = FRICTION_DYNAMIC

                    max_friction_force = friction_coeff * normal_force

                    # Apply friction by reducing tangential velocity
                    if tangent_speed > 1e-8:
                        tangent_dir = vel_tangent / tangent_speed

                        # Friction force opposes motion
                        # F = ma, so delta_v = F/m * dt
                        mass = 0.1
                        friction_decel = max_friction_force / mass * sub_dt

                        # Don't reverse direction - just slow down
                        new_tangent_speed = max(0, tangent_speed - friction_decel)
                        new_vel_tangent = tangent_dir * new_tangent_speed

                        # Update prev_pos to reflect new velocity (Verlet)
                        new_velocity = new_vel_tangent  # Normal component already handled by projection
                        node.prev_pos = node.pos - new_velocity * sub_dt
            else:
                node.in_contact = False

    def _calculate_normal_force(self, node_idx, normal):
        """Calculate the normal force pressing the node against the cylinder.

        This comes from the tension in adjacent segments pulling the node inward.
        """
        node = self.nodes[node_idx]
        normal_force = 0.0

        # Force from previous segment
        if node_idx > 0:
            prev_node = self.nodes[node_idx - 1]
            to_prev = prev_node.pos - node.pos
            dist = np.linalg.norm(to_prev)
            if dist > self.segment_length and dist > 1e-8:
                # Tension in this segment
                tension = CABLE_STIFFNESS * (dist - self.segment_length)
                direction = to_prev / dist
                # Component of tension force pointing inward (opposite to normal)
                inward_component = -np.dot(direction, normal) * tension
                if inward_component > 0:
                    normal_force += inward_component

        # Force from next segment
        if node_idx < len(self.nodes) - 1:
            next_node = self.nodes[node_idx + 1]
            to_next = next_node.pos - node.pos
            dist = np.linalg.norm(to_next)
            if dist > self.segment_length and dist > 1e-8:
                # Tension in this segment
                tension = CABLE_STIFFNESS * (dist - self.segment_length)
                direction = to_next / dist
                # Component of tension force pointing inward (opposite to normal)
                inward_component = -np.dot(direction, normal) * tension
                if inward_component > 0:
                    normal_force += inward_component

        return normal_force

    def _maybe_spawn_segment(self):
        """Spawn a new segment at the origin if the first free segment is too stretched.

        New nodes are positioned to keep the cable taut by maintaining proper segment lengths.
        """
        if len(self.nodes) < 2:
            return

        if len(self.nodes) >= MAX_SEGMENTS:
            return

        first_free = self.nodes[1]
        dist_to_origin = np.linalg.norm(first_free.pos - self.origin)

        # Only spawn if the first segment is stretched beyond threshold
        if dist_to_origin > self.segment_length * (1 + SPAWN_THRESHOLD):
            # Calculate direction from origin toward first free node
            direction = (first_free.pos - self.origin)
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            # Position new node at one segment length from the FIRST FREE NODE (not origin)
            # This keeps the cable taut by maintaining the constraint with the next node
            new_pos = first_free.pos - direction * self.segment_length

            new_node = CableNode(1, new_pos)
            new_node.prev_pos = new_pos.copy()  # Initialize prev_pos to avoid initial velocity

            # Insert after origin
            self.nodes.insert(1, new_node)

            # Renumber
            for i, node in enumerate(self.nodes):
                node.idx = i

            self.total_length += self.segment_length

    def get_node_positions(self):
        """Return array of all node positions for visualization."""
        return np.array([node.pos for node in self.nodes])

    def get_tension_at_origin(self):
        """Estimate tension at the cable origin.

        Since the first segment is a free pay-out segment, we measure tension
        from the second segment onwards.
        """
        if len(self.nodes) < 3:
            return 0.0

        # Measure tension in the second segment (first constrained segment)
        p1 = self.nodes[1].pos
        p2 = self.nodes[2].pos
        dist = np.linalg.norm(p2 - p1)
        stretch = max(0, dist - self.segment_length)
        # Approximate tension using virtual stiffness
        return CABLE_STIFFNESS * stretch


def add_visual_capsule(scene, p1, p2, radius, rgba):
    """Add a capsule between two points to the scene using mjv_connector."""
    if scene.ngeom >= scene.maxgeom:
        return

    # Validate inputs
    if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
        return

    length = np.linalg.norm(p2 - p1)
    if length < 1e-6 or not np.isfinite(length):
        return

    g = scene.geoms[scene.ngeom]

    # Use mjv_connector which properly creates a capsule between two points
    # Arguments: geom, type, width, from (3,1 array), to (3,1 array)
    from_pt = np.array(p1, dtype=np.float64).reshape(3, 1)
    to_pt = np.array(p2, dtype=np.float64).reshape(3, 1)

    mujoco.mjv_connector(
        g,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        float(radius),
        from_pt,
        to_pt
    )

    g.rgba[0] = rgba[0]
    g.rgba[1] = rgba[1]
    g.rgba[2] = rgba[2]
    g.rgba[3] = rgba[3]

    scene.ngeom += 1


def add_visual_sphere(scene, pos, radius, rgba):
    """Add a sphere to the scene using mjv_initGeom."""
    if scene.ngeom >= scene.maxgeom:
        return

    # Validate inputs
    if not np.all(np.isfinite(pos)):
        return

    g = scene.geoms[scene.ngeom]

    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0, 0]),  # size (only first element used for sphere)
        np.array(pos, dtype=np.float64),  # pos
        np.eye(3).flatten(),  # mat
        np.array(rgba, dtype=np.float32)  # rgba
    )

    scene.ngeom += 1


class Simulation:
    """Main simulation controller."""

    def __init__(self):
        self.xml = create_base_xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)

        self.cylinder_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_rotation")
        self.motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cylinder_motor")
        self.attachment_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_point")

        # Initial attachment point position (known from XML)
        initial_attachment = np.array([CYLINDER_RADIUS, 0.0, ATTACHMENT_HEIGHT])

        self.cable = CableSystem(
            origin=CABLE_ORIGIN,
            initial_attachment=initial_attachment,
            segment_length=CABLE_SEGMENT_LENGTH
        )

        self.time = 0.0
        self.wrap_count = 0.0

    def _get_attachment_world_pos(self):
        """Get world position of attachment point on cylinder."""
        site_xpos = self.data.site_xpos[self.attachment_site_id]
        return site_xpos.copy()

    def _get_cylinder_angle(self):
        """Get current cylinder rotation angle."""
        return self.data.qpos[self.cylinder_jnt_id]

    def step(self):
        """Advance simulation by one timestep."""
        self.data.ctrl[self.motor_id] = CYLINDER_OMEGA
        mujoco.mj_step(self.model, self.data)

        cylinder_angle = self._get_cylinder_angle()

        # Get the current attachment point position (rotates with cylinder)
        attachment_pos = self._get_attachment_world_pos()

        self.cable.update(self.model.opt.timestep, cylinder_angle, attachment_pos)

        self.wrap_count = cylinder_angle / (2 * np.pi)
        self.time = self.data.time

    def _get_attachment_world_pos(self):
        """Get world position of attachment point on cylinder."""
        # This returns the site position in world coordinates, which rotates with the cylinder
        return self.data.site_xpos[self.attachment_site_id].copy()

    def render_cable(self, scene):
        """Render cable segments into the scene."""
        positions = self.cable.get_node_positions()

        if len(positions) < 2:
            return

        # Origin marker (green)
        add_visual_sphere(scene, positions[0], 0.04, [0.0, 1.0, 0.0, 1.0])

        # Intermediate nodes - yellow if free, red if in contact with cylinder
        for i in range(1, len(positions) - 1):
            if self.cable.nodes[i].in_contact:
                color = [1.0, 0.0, 0.0, 1.0]  # Red for contact
            else:
                color = [1.0, 1.0, 0.0, 1.0]  # Yellow for free
            add_visual_sphere(scene, positions[i], 0.04, color)

        # Attachment point (orange)
        add_visual_sphere(scene, positions[-1], 0.04, [1.0, 0.5, 0.0, 1.0])

        # Cable segments
        for i in range(len(positions) - 1):
            p1 = positions[i]
            p2 = positions[i + 1]

            length = np.linalg.norm(p2 - p1)
            if length < 1e-6:
                continue

            # Color based on tension (stretch)
            stretch = max(0, length - self.cable.segment_length) / self.cable.segment_length
            tension_color = [
                min(1.0, 0.9 + stretch * 2),
                max(0.0, 0.2 - stretch),
                max(0.0, 0.1 - stretch),
                1.0
            ]

            add_visual_capsule(scene, p1, p2, CABLE_RADIUS, tension_color)

    def get_info_string(self):
        """Return string with current simulation info."""
        tension = self.cable.get_tension_at_origin()
        return (f"Time: {self.time:.2f}s | "
                f"Wraps: {self.wrap_count:.2f} | "
                f"Segments: {len(self.cable.nodes)} | "
                f"Cable Length: {self.cable.total_length:.1f}m | "
                f"Origin Tension: {tension:.0f}N")


def main():
    """Run the simulation with interactive viewer."""
    print("=" * 60)
    print("MuJoCo Cable Wrapping Simulation")
    print("=" * 60)
    print(f"Cylinder: radius={CYLINDER_RADIUS}m, height={CYLINDER_HEIGHT}m")
    print(f"Cable origin: {CABLE_ORIGIN}")
    print(f"Attachment point: {CYLINDER_RADIUS}m from axis, {ATTACHMENT_HEIGHT}m below origin")
    print(f"Rotation speed: {CYLINDER_OMEGA} rad/s")
    print("=" * 60)
    print("\nControls:")
    print("  - Close window or Ctrl+C to exit")
    print("  - Use mouse to rotate/zoom view")
    print("=" * 60)

    sim = Simulation()

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # Ensure proper aspect ratio
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 25
        viewer.cam.lookat[:] = [0, 0, 0]

        # Check if the cylinder (which we know is circular) also appears oval
        # If so, it's a viewer issue, not a cable rendering issue

        print("\nSimulation running... Close viewer window to stop.")
        print("NOTE: Check if the main cylinder also appears oval - if so, resize the window to be more square.")

        frame_count = 0
        while viewer.is_running():
            sim.step()

            # Render cable into user scene
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                sim.render_cable(viewer.user_scn)

            frame_count += 1
            if frame_count % 500 == 0:
                print(sim.get_info_string())
                print(f"  Attachment pos: {sim._get_attachment_world_pos()}")

            viewer.sync()

    print("\nSimulation ended.")
    print(f"Final state: {sim.get_info_string()}")


if __name__ == "__main__":
    main()