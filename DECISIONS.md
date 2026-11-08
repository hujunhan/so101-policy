# Decisions

Each entry: date, what was decided, why, and the options that were weighed.

## Open (to decide in S1)

- **The evaluation protocol**: number of trials, how initial states are drawn, the success criterion, the
  statistics (confidence intervals), the failure taxonomy; fixed and committed before any policy runs.
- **Cameras for the policy**: ZED left eye at FHD with a fixed crop, the wrist camera, or both; resolution and
  frame rate.
- **Recording**: LeRobot's own `lerobot-record` with the leader arm, or a thin wrapper through so101-core;
  where the dataset lives (Hugging Face Hub, public or private; local).
- **LeRobot version and policy extras**: the pin comes from so101-core; SmolVLA may need a newer LeRobot and
  extras.
- **The servo gain** for recording and evaluation: M2 ran at P = 32 (half the sag); LeRobot's default is 16.

## 2026-11-08: The task is one cube into a fixed bowl

**Decided:** pick one cube (about 3 cm, a solid matte colour) from a random position and yaw inside a region of
roughly 20 x 15 cm and place it in a bowl at a fixed position. The exact region and objects are fixed with the
protocol and the rebuilt workspace.

**Why:**
- Success is unambiguous (the cube rests in the bowl, released) and the reset is one object.
- The cube's position varies, so the policy must use the cameras.
- It is the most common SO-101 task: community reports put ACT with ~50 demonstrations at roughly 70-90 % in
  distribution, so there is something to compare against. A baseline below saturation leaves room to see M4's
  drops under camera changes and a gap between ACT and SmolVLA.
- M5's classical pipeline (detect in the ZED, transform, IK, top grasp) can do the same task.

**Options weighed:**
- *Target cube plus a distractor of another colour* (only the red one goes in): makes colour perception
  necessary, which would sharpen M4's white-balance tests; more failure modes, harder for ACT with 50 demos,
  two objects to reset.
- *Stacking one cube on another*: tests precision, but "stable stack" is less clear-cut and the classical
  pipeline in M5 would suffer from the few-mm errors far from the calibrated region.
- *Colour sorting into two boxes*: richer, but each demonstration and trial takes about twice as long.
