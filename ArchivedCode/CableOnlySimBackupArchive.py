"""
MuJoCo simulation of a discretized cable unspooling from a pulley.

Link behavior:
- All links start as TENSION-CONTROLLED (constant force, variable length)
- Once a link reaches maximum length, it switches to DISTANCE-CONTROLLED (stiff spring)
- This models cable being paid out from a spool segment by segment

The "active" tension-controlled link is always the one closest to the pulley
that hasn't yet reached max length.
"""

import mujoco
import mujoco.viewer
import numpy as np

# Simulation parameters
PULLEY_POS = np.array([0.0, 0.0, 2.0])
INITIAL_END_POS = np.array([0.5, 0.0, 2.0])  # Start closer to pulley
MOVE_DIRECTION = np.array([1.0, 0.0, 0.0])
MOVE_DIRECTION = MOVE_DIRECTION / np.linalg.norm(MOVE_DIRECTION)

# Cable parameters
NUM_SEGMENTS = 15
MAX_CABLE_LENGTH = 5.0
CONSTANT_TENSION = 10.0
SEGMENT_MASS = 0.05
SEGMENT_MAX_LENGTH = 0.25  # Max length before link switches to distance-controlled
CABLE_STIFFNESS = 5000.0
CABLE_DAMPING = 20.0

# Driving force
DRIVING_FORCE = CONSTANT_TENSION + 5.0


# Link state tracking
class LinkState:
    TENSION_CONTROLLED = 0
    DISTANCE_CONTROLLED = 1


def generate_model_xml():
    """Generate MuJoCo XML - tendons are visual only, forces applied manually."""

    initial_length = np.linalg.norm(INITIAL_END_POS - PULLEY_POS)
    segment_length = initial_length / NUM_SEGMENTS
    direction = (INITIAL_END_POS - PULLEY_POS) / initial_length

    xml = f"""
<mujoco model="dynamic_switching_cable">
    <option timestep="0.0005" gravity="0 0 0" integrator="implicit"/>

    <visual>
        <headlight diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    </visual>

    <default>
        <joint damping="0.5" armature="0.001"/>
    </default>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="512" height="512"/>
        <material name="grid_mat" texture="grid" texrepeat="5 5" reflectance="0.1"/>
        <material name="tension_mat" rgba="1.0 0.3 0.3 1"/>
        <material name="distance_mat" rgba="0.2 0.8 0.2 1"/>
        <material name="mass_mat" rgba="0.2 0.6 0.9 1"/>
        <material name="end_mat" rgba="0.9 0.9 0.2 1"/>
        <material name="pulley_mat" rgba="0.9 0.2 0.2 1"/>
    </asset>

    <worldbody>
        <geom type="plane" size="10 10 0.1" pos="0 0 0" material="grid_mat"/>

        <body name="pulley" pos="{PULLEY_POS[0]} {PULLEY_POS[1]} {PULLEY_POS[2]}">
            <geom type="sphere" size="0.08" material="pulley_mat"/>
            <site name="pulley_site" size="0.05"/>
        </body>
"""

    for i in range(NUM_SEGMENTS):
        pos = PULLEY_POS + direction * segment_length * (i + 1)
        is_end = (i == NUM_SEGMENTS - 1)

        xml += f"""
        <body name="seg_{i}" pos="{pos[0]} {pos[1]} {pos[2]}">
            <joint name="j{i}_x" type="slide" axis="1 0 0"/>
            <joint name="j{i}_y" type="slide" axis="0 1 0"/>
            <joint name="j{i}_z" type="slide" axis="0 0 1"/>
            <geom type="sphere" size="{"0.06" if is_end else "0.03"}" 
                  mass="{SEGMENT_MASS}" material="{"end_mat" if is_end else "mass_mat"}"/>
            <site name="site_{i}" size="0.02"/>
        </body>
"""

    # All tendons visual only - we control forces manually
    xml += """
    </worldbody>

    <tendon>
"""

    for i in range(NUM_SEGMENTS):
        prev_site = "pulley_site" if i == 0 else f"site_{i - 1}"
        xml += f"""
        <spatial name="tendon_{i}" width="0.02" rgba="0.5 0.5 0.5 1"
                 stiffness="0" damping="0">
            <site site="{prev_site}"/>
            <site site="site_{i}"/>
        </spatial>
"""

    xml += """
    </tendon>
</mujoco>
"""
    return xml, segment_length


# Very high stiffness for distance-controlled links
DISTANCE_STIFFNESS = 100000.0
DISTANCE_DAMPING = 500.0


class CableSimulation:
    def __init__(self):
        model_xml, self.initial_segment_length = generate_model_xml()
        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

        # Track state of each link (all start as tension-controlled)
        self.link_states = [LinkState.TENSION_CONTROLLED] * NUM_SEGMENTS
        self.link_fixed_lengths = [self.initial_segment_length] * NUM_SEGMENTS

        # Track when links switch (for logging)
        self.switch_times = [None] * NUM_SEGMENTS

    def get_segment_position(self, seg_idx):
        if seg_idx == -1:
            return PULLEY_POS.copy()
        return self.data.body(f"seg_{seg_idx}").xpos.copy()

    def get_link_length(self, link_idx):
        """Get length of link (link_idx connects seg_idx-1 to seg_idx)."""
        prev_pos = self.get_segment_position(link_idx - 1)
        curr_pos = self.get_segment_position(link_idx)
        return np.linalg.norm(curr_pos - prev_pos)

    def get_link_direction(self, link_idx):
        """Get unit vector from seg_idx toward seg_idx-1 (toward pulley)."""
        prev_pos = self.get_segment_position(link_idx - 1)
        curr_pos = self.get_segment_position(link_idx)
        diff = prev_pos - curr_pos
        dist = np.linalg.norm(diff)
        if dist > 1e-6:
            return diff / dist, dist
        return np.zeros(3), 0.0

    def get_total_cable_length(self):
        total = 0.0
        for i in range(NUM_SEGMENTS):
            total += self.get_link_length(i)
        return total

    def get_end_to_pulley_distance(self):
        end_pos = self.get_segment_position(NUM_SEGMENTS - 1)
        return np.linalg.norm(end_pos - PULLEY_POS)

    def update_link_states(self):
        """Check if any tension-controlled links should switch to distance-controlled."""
        for i in range(NUM_SEGMENTS):
            if self.link_states[i] == LinkState.TENSION_CONTROLLED:
                length = self.get_link_length(i)
                if length >= SEGMENT_MAX_LENGTH:
                    self.link_states[i] = LinkState.DISTANCE_CONTROLLED
                    self.link_fixed_lengths[i] = length
                    self.switch_times[i] = self.data.time

                    print(
                        f"  -> Link {i} switched to DISTANCE-CONTROLLED at t={self.data.time:.3f}s (fixed_length={length:.4f}m)")

    def find_active_link(self):
        """Find the first tension-controlled link (the one currently being paid out)."""
        for i in range(NUM_SEGMENTS):
            if self.link_states[i] == LinkState.TENSION_CONTROLLED:
                return i
        return None  # All links are distance-controlled

    def apply_link_forces(self):
        """
        Apply forces with proper tension propagation through the cable.

        Model:
        - Distance-controlled links: Very stiff spring to maintain fixed length
        - Active link: Tension-controlled (stretches freely under tension)
        - Links after active link: "On the spool" - stiff springs keep them bundled
        """
        self.data.xfrc_applied[:] = 0

        active_link = self.find_active_link()

        if active_link is None:
            active_link = NUM_SEGMENTS  # All distance-controlled

        # Process all links
        for i in range(NUM_SEGMENTS):
            direction, length = self.get_link_direction(i)

            # Get velocity for damping
            curr_vel = self.data.body(f"seg_{i}").cvel[3:]
            if i > 0:
                prev_vel = self.data.body(f"seg_{i - 1}").cvel[3:]
            else:
                prev_vel = np.zeros(3)
            rel_vel = np.dot(curr_vel - prev_vel, direction)

            if i < active_link:
                # Distance-controlled: very stiff spring to maintain fixed length
                stretch = length - self.link_fixed_lengths[i]
                spring_force = DISTANCE_STIFFNESS * stretch + DISTANCE_DAMPING * rel_vel
                force_magnitude = spring_force
            elif i == active_link:
                # Active link: just tension (stretches freely)
                force_magnitude = CONSTANT_TENSION
            else:
                # "On the spool": stiff spring to keep bundled with initial length
                stretch = length - self.initial_segment_length
                spring_force = CABLE_STIFFNESS * stretch + CABLE_DAMPING * rel_vel
                force_magnitude = spring_force

            # Apply force to current segment (toward previous/pulley)
            force = force_magnitude * direction
            body_id = self.model.body(f"seg_{i}").id
            self.data.xfrc_applied[body_id, :3] += force

            # Apply reaction force to previous segment (Newton's 3rd law)
            if i > 0:
                prev_body_id = self.model.body(f"seg_{i - 1}").id
                self.data.xfrc_applied[prev_body_id, :3] -= force

    def apply_driving_force(self):
        """Apply driving force to end mass."""
        end_idx = NUM_SEGMENTS - 1
        body_id = self.model.body(f"seg_{end_idx}").id
        self.data.xfrc_applied[body_id, :3] += DRIVING_FORCE * MOVE_DIRECTION

    def update_tendon_colors(self):
        """Update tendon colors based on link state (for visualization)."""
        for i in range(NUM_SEGMENTS):
            if self.link_states[i] == LinkState.TENSION_CONTROLLED:
                self.model.tendon_rgba[i] = [1.0, 0.3, 0.3, 1.0]  # Red
            else:
                self.model.tendon_rgba[i] = [0.2, 0.8, 0.2, 1.0]  # Green

    def count_states(self):
        """Count links in each state."""
        tension = sum(1 for s in self.link_states if s == LinkState.TENSION_CONTROLLED)
        distance = NUM_SEGMENTS - tension
        return tension, distance

    def step(self):
        """Perform one simulation step."""
        self.update_link_states()
        self.apply_link_forces()
        self.apply_driving_force()
        self.update_tendon_colors()
        mujoco.mj_step(self.model, self.data)


def run_simulation():
    """Main simulation loop with visualization."""
    sim = CableSimulation()

    initial_length = np.linalg.norm(INITIAL_END_POS - PULLEY_POS)

    print("=" * 70)
    print("Dynamic Link Switching Cable Simulation")
    print("=" * 70)
    print(f"Number of segments: {NUM_SEGMENTS}")
    print(f"Initial segment length: {sim.initial_segment_length:.4f} m")
    print(f"Maximum segment length (switch threshold): {SEGMENT_MAX_LENGTH:.4f} m")
    print(f"Initial total cable length: {initial_length:.3f} m")
    print(f"Maximum cable length: {MAX_CABLE_LENGTH:.3f} m")
    print("-" * 70)
    print("Link behavior:")
    print("  RED = Tension-controlled (constant force, can stretch)")
    print("  GREEN = Distance-controlled (stiff spring, fixed length)")
    print("-" * 70)
    print(f"Constant tension: {CONSTANT_TENSION:.1f} N")
    print(f"Spring stiffness (after switch): {CABLE_STIFFNESS:.0f} N/m")
    print("=" * 70)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        viewer.cam.distance = 8
        viewer.cam.lookat[:] = [2.5, 0, 1.5]

        simulation_running = True
        step_count = 0
        print_interval = 2000

        while viewer.is_running() and simulation_running:
            sim.step()
            step_count += 1

            import time
            time.sleep(0.1)

            if step_count % print_interval == 0:
                total_length = sim.get_total_cable_length()
                end_distance = sim.get_end_to_pulley_distance()
                tension_count, distance_count = sim.count_states()
                end_pos = sim.get_segment_position(NUM_SEGMENTS - 1)

                print(f"Time: {sim.data.time:6.3f}s | "
                      f"Length: {total_length:5.2f}m | "
                      f"Links: {tension_count}T/{distance_count}D | "
                      f"End: [{end_pos[0]:5.2f}, {end_pos[1]:5.2f}, {end_pos[2]:5.2f}]")

            if sim.get_end_to_pulley_distance() >= MAX_CABLE_LENGTH:
                print("\n" + "=" * 70)
                print("SIMULATION ENDED: Maximum cable length reached!")
                print(f"Final cable length: {sim.get_total_cable_length():.3f} m")
                print(f"Simulation time: {sim.data.time:.3f} s")
                tension_count, distance_count = sim.count_states()
                print(f"Final link states: {tension_count} tension-controlled, {distance_count} distance-controlled")
                print("=" * 70)
                simulation_running = False

                import time
                end_time = time.time()
                while viewer.is_running() and (time.time() - end_time) < 5.0:
                    viewer.sync()
                break

            viewer.sync()


def run_headless_simulation():
    """Run simulation without viewer."""
    sim = CableSimulation()

    print("Running headless simulation...")
    print(f"Segments: {NUM_SEGMENTS}, Max segment length: {SEGMENT_MAX_LENGTH}m")

    times, total_lengths, end_positions = [], [], []
    tension_counts, distance_counts = [], []
    segment_positions_history = []
    link_states_history = []

    max_time = 30.0

    while sim.data.time < max_time:
        sim.step()

        if len(times) == 0 or sim.data.time - times[-1] >= 0.02:
            times.append(sim.data.time)
            total_lengths.append(sim.get_total_cable_length())
            end_positions.append(sim.get_segment_position(NUM_SEGMENTS - 1))

            t, d = sim.count_states()
            tension_counts.append(t)
            distance_counts.append(d)

            seg_pos = [PULLEY_POS.copy()]
            for i in range(NUM_SEGMENTS):
                seg_pos.append(sim.get_segment_position(i))
            segment_positions_history.append(seg_pos)
            link_states_history.append(sim.link_states.copy())

        if sim.get_end_to_pulley_distance() >= MAX_CABLE_LENGTH:
            print(f"Max length reached at t={sim.data.time:.3f}s")
            break

    return (np.array(times), np.array(total_lengths), np.array(end_positions),
            np.array(tension_counts), np.array(distance_counts),
            segment_positions_history, link_states_history)


def visualize_results(times, total_lengths, end_positions, tension_counts,
                      distance_counts, segment_history, states_history):
    """Plot simulation results."""
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14, 10))

        # Cable length over time
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(times, total_lengths, 'b-', linewidth=2)
        ax1.axhline(y=MAX_CABLE_LENGTH, color='r', linestyle='--', label='Max length')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Cable Length (m)')
        ax1.set_title('Total Cable Length vs Time')
        ax1.legend()
        ax1.grid(True)

        # Link states over time
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.fill_between(times, 0, distance_counts, color='green', alpha=0.7, label='Distance-controlled')
        ax2.fill_between(times, distance_counts, distance_counts + tension_counts,
                         color='red', alpha=0.7, label='Tension-controlled')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Number of Links')
        ax2.set_title('Link States Over Time')
        ax2.legend()
        ax2.set_ylim(0, NUM_SEGMENTS)
        ax2.grid(True)

        # 3D cable snapshots
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        n_snapshots = min(5, len(segment_history))
        indices = np.linspace(0, len(segment_history) - 1, n_snapshots, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))

        for idx, color in zip(indices, colors):
            seg_pos = np.array(segment_history[idx])
            ax3.plot(seg_pos[:, 0], seg_pos[:, 1], seg_pos[:, 2],
                     'o-', color=color, markersize=3, linewidth=1.5,
                     label=f't={times[idx]:.2f}s')

        ax3.scatter(*PULLEY_POS, color='red', s=100, marker='o')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('Cable Configuration Over Time')
        ax3.legend(loc='upper left', fontsize=8)

        # Final cable with color-coded links
        ax4 = fig.add_subplot(2, 2, 4)
        final_seg = np.array(segment_history[-1])
        final_states = states_history[-1]

        for i in range(NUM_SEGMENTS):
            color = 'red' if final_states[i] == LinkState.TENSION_CONTROLLED else 'green'
            ax4.plot(final_seg[i:i + 2, 0], final_seg[i:i + 2, 2], 'o-',
                     color=color, markersize=6, linewidth=3)

        ax4.plot(PULLEY_POS[0], PULLEY_POS[2], 'ko', markersize=12, label='Pulley')
        ax4.plot([], [], 'r-', linewidth=3, label='Tension-controlled')
        ax4.plot([], [], 'g-', linewidth=3, label='Distance-controlled')
        ax4.set_xlabel('X Position (m)')
        ax4.set_ylabel('Z Position (m)')
        ax4.set_title('Final Cable Configuration')
        ax4.legend()
        ax4.grid(True)
        ax4.axis('equal')

        plt.tight_layout()
        plt.savefig('dynamic_switching_cable_results.png', dpi=150)
        plt.show()
        print("Results saved to dynamic_switching_cable_results.png")

    except ImportError:
        print("Matplotlib not available for plotting")


"""
# Debug: print detailed info at specific times
debug_times = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]
# should_debug = any(abs(self.time - t) < 0.003 for t in debug_times)
should_debug = False

if should_debug:
    print(f"\n=== TENSION DEBUG t={self.time:.3f}s ===")
    print(f"active_count={self.active_count}, spool_idx={self.spool_idx}")

    # Check spool position
    spool_pos = self._get_node_pos(self.spool_idx)
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

'''
def _debug_force_balance(self):
    """Debug the force balance on the spool body."""
    spool_body_id = self.cable_body_ids[self.spool_idx]
    spool_pos = self._get_node_pos(self.spool_idx)

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
    prev_pos = self._get_node_pos(self.spool_idx - 1)
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
            pos = self._get_node_pos(idx)
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
            pos = self._get_node_pos(idx)
            vel = self.data.cvel[self.cable_body_ids[idx], 3:6]
            dist_from_origin = pos[0] - CABLE_ORIGIN[0]
            print(f"  SPAWN2 t+{elapsed:.3f}s: pos_x={pos[0]:.4f}, dist={dist_from_origin:.4f}, vel_x={vel[0]:.4f}")
'''

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--headless":
        results = run_headless_simulation()
        visualize_results(*results)
    else:
        run_simulation()