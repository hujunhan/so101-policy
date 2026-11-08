# Decisions

Each entry: date, what was decided, why, and the options that were weighed.

## Open (to decide in S1)

- **Cameras for the policy**: ZED left eye at FHD with a fixed crop, the wrist camera, or both; resolution and
  frame rate.
- **Where the dataset lives**: Hugging Face Hub (public or private) or local only.
- **LeRobot version and policy extras**: the pin comes from so101-core; SmolVLA may need a newer LeRobot and
  extras.
- **The servo gain** for recording and evaluation: M2 ran at P = 32 (half the sag); LeRobot's default is 16.

## 2026-11-08: Record through `so101-record`, a thin wrapper around `lerobot-record`

**Decided:** demonstrations are recorded with LeRobot's own record loop (leader arm, LeRobot dataset format),
started through `so101-record`. The wrapper fills in the command from so101-core's `devices.yaml` and the
committed `protocol/recording.yaml` (cameras, crop, V4L2 controls, servo gain, step cap), registers a "zed"
camera type (left eye of the side-by-side stream, fixed crop, resize), checks that the leader is in the
follower's pose before starting, and syncs the follower's goal to its present position.

**Why:**
- The dataset is in LeRobot's standard format, so ACT and SmolVLA train on it without conversion.
- The protocol requires the same camera settings and gain at evaluation; one committed settings file
  instead of a 20-argument command typed by hand makes that checkable.
- LeRobot's OpenCV camera cannot crop, and its follower `connect()` can jump to a stale goal (M0 findings);
  both are handled in the wrapper without changing LeRobot.

**Options weighed:**
- *Plain `lerobot-record`*: no code to maintain, but long commands, no ZED crop, no start-pose check.
- *A recording loop of our own through `SafeArm`*: full control, but rewrites what LeRobot already does
  (dataset writing, video encoding, keyboard control) and risks a non-standard dataset.

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

## 2026-11-08: The evaluation protocol

**Decided** (full text in [docs/PROTOCOL.md](docs/PROTOCOL.md)):
- **30 trials** per policy and condition.
- **Initial states from a seeded random list** (seed 20261108, uniform position in the region, yaw in
  [0, 90) deg), committed as `protocol/eval_states.csv`; the same list for every policy and condition in M3,
  M4 and M5.
- **Time limit by rule**: 3 x the median duration of S2's dry-run demonstrations, fixed before any policy runs.
- **Statistics**: 95 % Wilson interval per rate; exact McNemar test for paired comparisons.
- **Discipline**: final checkpoint, no retries, voids only for faults outside the policy, a fixed failure
  taxonomy labelled from video (first thing that went wrong).

**Why:**
- *30 trials*: about +/-14 points at 80 % success (20 gives +/-17, 50 gives +/-11). M4 runs six camera
  conditions, so 50 per condition would be close to 6 hours of trials for 3 points of width.
- *A shared random list*: random draws do not line up with habits from the demonstrations, as a regular grid
  might; using the same list everywhere makes comparisons paired, and the paired test only looks at trials where
  the two policies disagree, so it detects a real difference with fewer trials.
- *Time limit by rule*: the demonstrations' length is not known yet; fixing the rule now and the number before
  the first policy run keeps it independent of how the policies behave.
- *Final checkpoint*: choosing the checkpoint on the evaluation list would overstate the success rate.

**Options weighed:** 20 or 50 trials; a fixed 5 x 6 grid of positions (easier to place, but regular); a fixed
time limit set now (e.g. 60 s).
