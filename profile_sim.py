"""Profile the simulation for the first 5 seconds of sim time, including rendering."""
import cProfile
import pstats
import io
from time import sleep

import mujoco
import mujoco.viewer

from GeckoLassoSim import Simulation

def run_5s():
    sim = Simulation()

    with mujoco.viewer.launch_passive(sim.display_model, sim.display_data) as viewer:
        viewer.cam.azimuth = -0.8289683948863515
        viewer.cam.elevation = -21.25310724431816
        viewer.cam.distance = 34.68901593838663
        viewer.cam.lookat[:] = [0, 5, 0]
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        sleep(0.5)

        import time as _time
        render_interval = 1.0 / 60.0
        last_render_wall = _time.monotonic()

        while viewer.is_running() and sim.data.time < 5.0:
            now = _time.monotonic()
            while now - last_render_wall < render_interval:
                sim.step()
                now = _time.monotonic()

            last_render_wall = now

            node_positions, node_colors = sim.sync_display()

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                sim.render_cable(viewer.user_scn, node_positions, node_colors)
                sim.render_overlay(viewer.user_scn, viewer)

            viewer.sync()

    print(f"Finished at sim time {sim.data.time:.4f}s, "
          f"active_count={sim.active_count}, anchor_idx={sim.anchor_idx}")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run_5s()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(40)
    print(stream.getvalue())

    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats("tottime")
    stats2.print_stats(40)
    print("\n=== Sorted by total time ===")
    print(stream2.getvalue())
