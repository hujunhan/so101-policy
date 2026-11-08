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

> **Status (2026-11-08):** S1 done (task, pre-registered protocol). S2 pipeline working (`so101-record`,
> `so101-check-dataset`, `so101-replay`). S4 dry run on a public dataset done: ACT trains in ~2 h for 30k steps;
> 20k steps were enough there. Waiting for the light box: relocate the ZED (`so101-handeye relocate`), the ZED
> crop, the camera settings and PROTOCOL section 7, then S3.

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

## S4 dry run on a public dataset (2026-11-08)

ACT (LeRobot defaults: chunk 100, ResNet18, CVAE) on `lerobot/svla_so101_pickplace` (50 SO-101 pick-and-place
episodes, two 640 x 480 cameras), the last 5 episodes held out. Batch 8, bf16 mixed precision, 30k steps
(~23 epochs): 1 h 54 min on the RTX 3080 Laptop, 0.22 s per step (0.29 s in fp32), 3.6 GB of GPU memory;
data loading is not the bottleneck (0.003 s per step).

- **Training loss keeps falling** (L1 0.28 at 2k steps, 0.09 at 30k); **the held-out loss is flat from 2k
  steps on** (0.36 -> 0.34). With 45 demonstrations most of what the later steps learn does not carry over to
  new episodes. (The two are not directly comparable: training uses the CVAE's latent from the true actions,
  held-out uses the zero latent as at deployment.)
- **Open-loop action error on the held-out episodes** (`so101-offline-eval`; mean over the actions a deployed
  policy would execute, deg; gripper %), against "hold still" (predict the current state):

  | Checkpoint | pan | lift | elbow | wrist flex | wrist roll | gripper |
  |---|---:|---:|---:|---:|---:|---:|
  | hold still | 51.4 | 25.3 | 21.5 | 10.2 | 22.7 | 9.4 |
  | 10k steps | 28.2 | 15.6 | 11.5 | 6.9 | 11.4 | 4.7 |
  | 20k steps | 24.4 | 12.5 | 10.4 | 6.4 | 9.6 | 3.9 |
  | 30k steps | 26.0 | 12.4 | 10.1 | 6.3 | 9.8 | 3.8 |

  10k -> 20k helps by 10-20 %; 20k -> 30k does not (within the noise of 5 episodes). The predicted motion has
  the right shape (reach, grasp, lift, place, return), and much of the remaining error is **timing**: the
  policy starts about 1 s before a demonstrator who waited 2 s. Open-loop error is a check of training, not of
  task success: on the arm the policy reacts to what it sees.
- For our recordings: start moving soon after "Recording episode" and stop soon after returning, so the
  demonstrations do not teach a random wait.

## Hardware

- The SO-101 follower and leader, as in M0-M2 (device paths in so101-core's git-ignored `config/devices.yaml`).
- ZED 2 on a fixed mount, read as a UVC stereo camera (factory calibration); wrist camera on the gripper (USB,
  1280 x 720 MJPG, fixed focus).
- Linux laptop with an RTX 3080 Laptop GPU (16 GB).
- Workspace being rebuilt: light box and backdrop.

## Decisions

See [DECISIONS.md](DECISIONS.md).
