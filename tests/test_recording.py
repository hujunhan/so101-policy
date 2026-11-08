"""Recording pipeline without hardware: the ZED crop, the camera config, the LeRobot command line."""

import argparse
import json

import numpy as np
import pytest

pytest.importorskip("lerobot", reason="install with `uv sync --extra hw`")

from so101_policy.record import camera_configs, lerobot_args, load_settings, mask_serials  # noqa: E402
from so101_policy.zed_camera import ZedCamera, ZedCameraConfig, crop_left_eye  # noqa: E402


def side_by_side(w_eye=8, h=4):
    """Left eye filled with 1s, right eye with 2s; pixel (y, x) of the left eye also encodes its position."""
    frame = np.full((h, 2 * w_eye, 3), 2, dtype=np.uint8)
    frame[:, :w_eye] = 1
    frame[:, :w_eye, 0] = np.arange(h)[:, None] * 10 + np.arange(w_eye)[None, :]
    return frame


def test_crop_takes_the_left_eye_and_the_roi():
    f = side_by_side()
    out = crop_left_eye(f, [2, 1, 3, 2], None)
    assert out.shape == (2, 3, 3)
    assert (out[..., 1] == 1).all()  # left eye only
    assert out[0, 0, 0] == 1 * 10 + 2 and out[1, 2, 0] == 2 * 10 + 4
    assert out.flags["C_CONTIGUOUS"]


def test_crop_resizes():
    f = np.zeros((1080, 3840, 3), np.uint8)
    assert crop_left_eye(f, [0, 0, 1920, 1080], (960, 540)).shape == (540, 960, 3)


def zed_cfg(**kw):
    base = dict(index_or_path="/dev/null", fps=30, width=3840, height=1080, fourcc="YUYV")
    return ZedCameraConfig(**{**base, **kw})


def test_config_defaults_to_the_whole_left_eye():
    cfg = zed_cfg()
    assert cfg.crop == [0, 0, 1920, 1080]
    assert cfg.output_size == (1920, 1080)
    assert zed_cfg(crop=[100, 50, 800, 600]).output_size == (800, 600)
    assert zed_cfg(out_width=640, out_height=360).output_size == (640, 360)


@pytest.mark.parametrize(
    "kw",
    [
        dict(crop=[1500, 0, 500, 100]),  # runs past the left eye into the right one
        dict(crop=[0, 0, 10]),
        dict(out_width=640),
        dict(width=None, height=None),
    ],
)
def test_bad_configs_are_refused(kw):
    with pytest.raises(ValueError):
        zed_cfg(**kw)


def test_camera_reports_the_output_size_before_connecting():
    # LeRobot builds the dataset features from cam.width/height before it connects the cameras.
    cam = ZedCamera(zed_cfg(crop=[100, 50, 800, 600], out_width=400, out_height=300))
    assert (cam.width, cam.height) == (400, 300)
    assert (cam.capture_width, cam.capture_height) == (3840, 1080)


def test_postprocess_crops_and_converts_to_rgb():
    cam = ZedCamera(zed_cfg(crop=[0, 0, 4, 2]))
    frame = np.zeros((1080, 3840, 3), np.uint8)
    frame[0, 0] = [255, 0, 0]  # BGR blue
    out = cam._postprocess_image(frame)
    assert out.shape == (2, 4, 3)
    assert out[0, 0].tolist() == [0, 0, 255]  # RGB blue
    with pytest.raises(RuntimeError):
        cam._postprocess_image(np.zeros((720, 2560, 3), np.uint8))


class Devices:
    follower_port = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_ABC123XYZ-if00"
    leader_port = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_DEF456UVW-if00"
    robot_id = "so101"
    teleop_id = "so101"
    cameras = {
        "zed": "/dev/zed",
        "wrist": "/dev/v4l/by-id/usb-Generic_USB_HD_Camera_123456789012-video-index0",
    }


def test_committed_settings_build_a_valid_command():
    settings = load_settings()
    cams = camera_configs(settings, Devices.cameras)
    assert cams["zed"]["type"] == "zed" and cams["wrist"]["type"] == "opencv"
    ZedCameraConfig(**{k: v for k, v in cams["zed"].items() if k != "type"})  # the ZED part parses

    args = argparse.Namespace(
        repo_id="local/x", root=None, task="Put the cube in the bowl.", episodes=2,
        episode_time=20.0, reset_time=10.0, display=False, resume=False,
    )  # fmt: skip
    cmd = lerobot_args(settings, Devices, args)
    opts = dict(a.removeprefix("--").split("=", 1) for a in cmd)
    assert opts["robot.type"] == "so101_follower" and opts["teleop.type"] == "so101_leader"
    assert opts["dataset.push_to_hub"] == "false"
    assert json.loads(opts["robot.cameras"]) == cams
    assert opts["dataset.root"].endswith("data/local__x")


def test_serials_are_masked():
    text = " ".join([Devices.follower_port, Devices.cameras["wrist"]])
    masked = mask_serials(text)
    assert "ABC123XYZ" not in masked and "123456789012" not in masked
    assert masked.count("<SERIAL>") == 2


def test_start_pose_check_names_the_far_joints():
    from so101_policy.record import pose_differences

    leader = {"wrist_roll": 65.1, "elbow_flex": 90.0, "gripper": 30.0}
    follower = {"wrist_roll": -92.3, "elbow_flex": 85.0, "gripper": 5.0}
    far = pose_differences(leader, follower, max_deg=15.0, max_grip=20.0)
    assert [j for j, _, _ in far] == ["wrist_roll", "gripper"]
    assert pose_differences(leader, leader, 15.0, 20.0) == []
