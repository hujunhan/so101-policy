# so101-policy

M3 of the [SO-101 series](https://github.com/hujunhan/so101-core). It teaches the arm one manipulation task from
teleoperated demonstrations and measures how well learned policies do it:

- **One task**, chosen so success is unambiguous and the scene can be reset quickly.
- **50-100 teleoperated demonstrations** with the leader arm, recorded through LeRobot (ZED 2 scene view and
  wrist camera).
- **ACT, then a fine-tuned SmolVLA** (pi0.5 as a stretch).
- **A pre-registered evaluation protocol**: the trials, the success criterion and the analysis fixed before
  any policy is run; success rates with confidence intervals and a failure taxonomy.

Deliverable: a video, a results table and a failure analysis.

> **Status (2026-11-08):** repository set up; session plan agreed. Next: S1, the task and the evaluation protocol.

## What M2 hands over

From [so101-calib](https://github.com/hujunhan/so101-calib) (README, "Handed to M3 and M5"):

- **Calibration in hand**: the elbow's zero offset (7.3 deg) is in so101-core's joint map; LeRobot's own
  calibration and recordings are unchanged. `results/s4/handeye.yaml` has the wrist camera on the gripper and
  the ZED in the base frame; `results/s1/servo_model.json` the servos' static model.
- **The ZED pose is tied to the current setup.** The workspace is being rebuilt (light box, backdrop); after
  that the ZED has to be located again with a short hand-eye re-run (not written yet).
- **The ZED sees ~100 deg**, mostly room: plan FHD per eye and a fixed crop around the workspace. The background
  behind the arm stays visible in any light box: keep it constant between training and evaluation, keep people
  and animals out of view during evaluation.
- **Accuracy**: near the calibrated region ~1-2 mm per error term; far poses several mm off (cause open).
  Learned policies close the loop through the cameras, so this matters less here than for M5.

## Plan

| Session | Content | Output |
|---|---|---|
| **S1** | Choose the task and write the evaluation protocol (success criterion, trial count, initial-state distribution, analysis) before any data | `docs/PROTOCOL.md`, DECISIONS |
| **S2** | Recording pipeline: LeRobot record with the leader arm, ZED (FHD, cropped) and wrist camera; a few dry-run episodes; replay and check | Pipeline, a test dataset |
| **S3** | Collect 50-100 demonstrations in the final environment | Dataset (Hugging Face Hub or local) |
| **S4** | Train ACT on the laptop GPU; offline checks (loss, action replay) | Checkpoint, training report |
| **S5** | Evaluate ACT by the protocol | Success rate with CI, failure taxonomy |
| **S6** | Fine-tune SmolVLA; evaluate by the same protocol | Comparison table |
| **S7** | Write-up: README, `docs/DESIGN.md`, `docs/THEORY.md`, video | |

S1, S2 and S4 (on a public dataset) can start before the environment is rebuilt; S3 and the evaluations need
the final environment, because policies are sensitive to background and lighting.

## Hardware

- The SO-101 follower and leader, as in M0-M2 (device paths in so101-core's git-ignored `config/devices.yaml`).
- ZED 2 on a fixed mount, read as a UVC stereo camera (factory calibration); wrist camera on the gripper (USB,
  1280 x 720 MJPG, fixed focus).
- Linux laptop with an RTX 3080 Laptop GPU (16 GB).
- Workspace being rebuilt: light box and backdrop.

## Decisions

See [DECISIONS.md](DECISIONS.md).
