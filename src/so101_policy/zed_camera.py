"""The ZED 2 as a LeRobot camera: left eye of the side-by-side UVC stream, a fixed crop, an optional resize.

LeRobot's OpenCVCamera has no crop, and it checks every frame against the configured width and height. Here
the config's `width`/`height` are the *capture* size (the whole side-by-side frame, e.g. 3840x1080 for FHD);
the camera reports the *output* size (crop, then `out_width` x `out_height`) as its `width`/`height`, which is
what LeRobot stores in the dataset features.

Registered as camera type "zed" when this module is imported (`so101-record` does that before parsing).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from lerobot.cameras import CameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


@CameraConfig.register_subclass("zed")
@dataclass
class ZedCameraConfig(OpenCVCameraConfig):
    # Crop in left-eye pixels: x, y, width, height. None means the whole left eye.
    crop: list[int] | None = None
    # Output size after the crop; None keeps the crop size.
    out_width: int | None = None
    out_height: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.width is None or self.height is None:
            raise ValueError("ZedCameraConfig needs the capture width and height (e.g. 3840x1080 for FHD).")
        if self.width % 2:
            raise ValueError(f"Side-by-side width must be even, got {self.width}.")
        if self.rotation != Cv2Rotation.NO_ROTATION:
            raise ValueError("Rotation is not supported for the ZED.")
        eye_w, eye_h = self.width // 2, self.height
        if self.crop is None:
            self.crop = [0, 0, eye_w, eye_h]
        if len(self.crop) != 4:
            raise ValueError(f"crop must be [x, y, w, h], got {self.crop}.")
        x, y, w, h = self.crop
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > eye_w or y + h > eye_h:
            raise ValueError(f"crop {self.crop} does not fit in the left eye ({eye_w}x{eye_h}).")
        if (self.out_width is None) != (self.out_height is None):
            raise ValueError("Give both out_width and out_height, or neither.")

    @property
    def output_size(self) -> tuple[int, int]:
        """(width, height) of the frames the camera returns."""
        if self.out_width is not None and self.out_height is not None:
            return self.out_width, self.out_height
        return self.crop[2], self.crop[3]


def crop_left_eye(frame: np.ndarray, crop: list[int], out_size: tuple[int, int] | None) -> np.ndarray:
    """Left half of a side-by-side frame, cropped to [x, y, w, h], resized to out_size (w, h) if given."""
    x, y, w, h = crop
    eye = frame[:, : frame.shape[1] // 2]
    roi = eye[y : y + h, x : x + w]
    if out_size is not None and out_size != (w, h):
        # INTER_AREA: the standard choice for downscaling (averages, no aliasing).
        return cv2.resize(roi, out_size, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(roi)


class ZedCamera(OpenCVCamera):
    def __init__(self, config: ZedCameraConfig):
        self.zed_config = config
        super().__init__(config)

    def __str__(self) -> str:
        return f"ZedCamera({self.index_or_path})"

    def _reset_connection_settings(self) -> None:
        super()._reset_connection_settings()
        # Capture the whole side-by-side frame; report the output size to LeRobot.
        self.capture_width, self.capture_height = self.config.width, self.config.height
        self.width, self.height = self.zed_config.output_size

    def _configure_capture_settings(self) -> None:
        # The parent checks `self.width/height is None` and then sets capture size from capture_width/height.
        super()._configure_capture_settings()
        self.width, self.height = self.zed_config.output_size

    def _postprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Crop first (the parent checks the full capture size), then let the parent convert colour.
        h, w = image.shape[:2]
        if (w, h) != (self.capture_width, self.capture_height):
            raise RuntimeError(
                f"{self} frame {w}x{h} does not match capture {self.capture_width}x{self.capture_height}."
            )
        out = crop_left_eye(image, self.zed_config.crop, self.zed_config.output_size)
        if self.color_mode == ColorMode.RGB:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out
