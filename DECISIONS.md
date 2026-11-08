# Decisions

Each entry: date, what was decided, why, and the options that were weighed.

## Open (to decide in S1)

- **The task**: what the arm does, and what counts as success (unambiguous, quick to reset, needs the cameras).
- **The evaluation protocol**: number of trials, how initial states are drawn, the success criterion, the
  statistics (confidence intervals), the failure taxonomy; fixed and committed before any policy runs.
- **Cameras for the policy**: ZED left eye at FHD with a fixed crop, the wrist camera, or both; resolution and
  frame rate.
- **Recording**: LeRobot's own `lerobot-record` with the leader arm, or a thin wrapper through so101-core;
  where the dataset lives (Hugging Face Hub, public or private; local).
- **LeRobot version and policy extras**: the pin comes from so101-core; SmolVLA may need a newer LeRobot and
  extras.
- **The servo gain** for recording and evaluation: M2 ran at P = 32 (half the sag); LeRobot's default is 16.
