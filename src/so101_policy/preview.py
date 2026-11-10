"""A light live view for `so101-record --display`: both camera images as stored, and the joints.

LeRobot's own display (rerun) stopped responding on this laptop, so `so101-record` replaces LeRobot's
visualization hooks with this OpenCV window. It shows the frames exactly as they go into the dataset (the ZED
cropped and resized, the wrist camera resized), the follower's joints and the leader's commands, and updates
at up to 15 Hz in its own thread, so the 30 Hz control loop does not wait for it.

The keys of the recording loop also work in the window (n or Right = end the episode or the reset, r or
Left = redo the episode, q or Esc = stop), so it does not matter whether the window or the terminal has focus.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

WINDOW = "so101-record: what the policy sees"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ROW_H = 360  # both images are shown at this height
KEYS = {ord("n"): "right", 83: "right", ord("r"): "left", 81: "left", ord("q"): "esc", 27: "esc"}


def compose(observation: dict, action: dict | None, row_h: int = ROW_H) -> np.ndarray:
    """One BGR image: the camera images side by side at row_h, then a strip with the joint values."""
    tiles = []
    for key, img in observation.items():
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3:
            w = int(round(img.shape[1] * row_h / img.shape[0]))
            tile = cv2.resize(img, (w, row_h), interpolation=cv2.INTER_AREA)
            tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            label = f"{key} {img.shape[1]}x{img.shape[0]}"
            cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(tile)
    top = np.hstack(tiles) if tiles else np.zeros((row_h, 640, 3), np.uint8)
    strip = np.zeros((58, top.shape[1], 3), np.uint8)
    col = top.shape[1] // len(JOINTS)
    for i, j in enumerate(JOINTS):
        s = observation.get(f"{j}.pos")
        a = None if action is None else action.get(f"{j}.pos")
        x = 8 + i * col
        cv2.putText(strip, j, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        text = ("-" if s is None else f"{float(s):6.1f}") + ("" if a is None else f" / {float(a):6.1f}")
        cv2.putText(strip, text, (x, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(strip, "follower / leader (deg; gripper %)", (top.shape[1] - 330, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)  # fmt: skip
    return np.vstack([top, strip])


class Preview:
    """The control loop only hands over its latest observation (no copy, no drawing); a background thread
    draws the window at up to max_hz, so the loop never waits for the GUI. All OpenCV GUI calls happen in
    that one thread."""

    def __init__(self, max_hz: float = 15.0):
        self.period = 1.0 / max_hz
        self.latest: tuple[dict, dict | None] | None = None
        self.events: dict | None = None
        self.apply = None
        self.stop_flag = threading.Event()
        self.thread: threading.Thread | None = None

    def show(self, observation: dict | None = None, action: dict | None = None, **_) -> None:
        if observation is None:
            return
        self.latest = (observation, action)
        if self.thread is None:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self) -> None:
        shown = None
        while not self.stop_flag.is_set():
            t = time.perf_counter()
            item = self.latest
            if item is not None and item is not shown:
                cv2.imshow(WINDOW, compose(*item))
                shown = item
            key = cv2.waitKey(1) & 0xFF
            if key in KEYS and self.events is not None and self.apply is not None:
                self.apply(KEYS[key], self.events)
            time.sleep(max(0.0, self.period - (time.perf_counter() - t)))
        # No destroyWindow: Qt warns when its window is torn down outside the main thread; the window
        # closes when the process exits right after.

    def close(self) -> None:
        self.stop_flag.set()
        if self.thread is not None:
            self.thread.join(timeout=2)


def install(record_module, preview: Preview) -> None:
    """Swap lerobot_record's visualization hooks for the preview, and share the keyboard events with it."""
    from lerobot.utils.keyboard_input import apply_recording_control

    original_listener = record_module.init_keyboard_listener

    def init_keyboard_listener():
        listener, events = original_listener()
        preview.events, preview.apply = events, apply_recording_control
        return listener, events

    record_module.init_keyboard_listener = init_keyboard_listener
    record_module.init_visualization = lambda *a, **k: None
    record_module.log_visualization_data = lambda mode, **k: preview.show(**k)
    record_module.shutdown_visualization = lambda *a, **k: preview.close()
