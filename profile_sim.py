"""Profile the simulation for the first 15 seconds of sim time, including rendering and screen recording.

Run with: mjpython profile_sim.py
"""
import cProfile
import pstats
import io
import os
from time import sleep
from threading import Thread, Event

import numpy as np
import mujoco
import mujoco.viewer
import Quartz.CoreGraphics as CG
import imageio.v3 as iio

from GeckoLassoSim import Simulation, RECORD_FPS

RECORD_OUTPUT = "profile_recording.mp4"

SIM_DURATION = 15.0

def run_profiled():
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

        # Recording setup — capture blocks main thread, processing on background thread
        record_dt = 1.0 / RECORD_FPS
        next_record_time = 0.0
        frame_count = 0
        capture_event = Event()
        done_event = Event()
        stop_event = Event()
        _shared_cg_image = [None]

        def _record_worker():
            writer = iio.imopen(RECORD_OUTPUT, "w", plugin="pyav")
            writer.init_video_stream("libx264", fps=RECORD_FPS)
            while True:
                capture_event.wait()
                capture_event.clear()
                if stop_event.is_set():
                    break
                _shared_cg_image[0] = CG.CGWindowListCreateImage(
                    CG.CGRectInfinite, CG.kCGWindowListOptionOnScreenOnly,
                    CG.kCGNullWindowID, CG.kCGWindowImageDefault)
                done_event.set()
                cg_image = _shared_cg_image[0]
                w = CG.CGImageGetWidth(cg_image)
                h = CG.CGImageGetHeight(cg_image)
                raw = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(cg_image))
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
                frame = frame[:, :, :3][:, :, ::-1].copy()
                frame = frame[:h - h % 2, :w - w % 2]
                writer.write_frame(frame)
            writer.close()

        record_thread = Thread(target=_record_worker, daemon=True)
        record_thread.start()

        try:
            while viewer.is_running() and sim.data.time < SIM_DURATION:
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

                # Trigger capture on background thread, block until screenshot is taken
                if sim.data.time >= next_record_time:
                    capture_event.set()
                    done_event.wait()
                    done_event.clear()
                    frame_count += 1
                    next_record_time += record_dt
        finally:
            stop_event.set()
            capture_event.set()
            record_thread.join()

    # Clean up the test recording
    if os.path.exists(RECORD_OUTPUT):
        os.remove(RECORD_OUTPUT)

    print(f"\nFinished at sim time {sim.data.time:.4f}s, "
          f"active_count={sim.active_count}, anchor_idx={sim.anchor_idx}, "
          f"frames_recorded={frame_count}")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run_profiled()
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
