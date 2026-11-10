# Decisions

Each entry: date, what was decided, why, and the options that were weighed.

## Open

Nothing open.

## 2026-11-09: Datasets stay local

**Decided:** recorded datasets stay on this laptop in the git-ignored `data/` (LeRobot v3 format, `local/...`
repo ids; `so101-record` never pushes). Back up `data/` to another disk after each recording session.

**Why:** training reads the local files directly; nothing has to be uploaded or downloaded. The videos show
the light box and the arm only, but every episode would need checking for people or the room before it went
public. Whether to publish the final dataset is decided at S7.

**Options:** local only (chosen); Hugging Face Hub, private (a cloud copy); Hub, public (reproducible by
others; needs a privacy review of the videos).

## 2026-11-09: Servo P gain 16 for recording and evaluation

**Decided:** the follower runs at P = 16 (LeRobot's default) in demonstrations and in policy runs.

**Why:** the S2 pipeline was verified at 16 (a slow replay repeated the recorded joint states within 0.1-0.4 deg
median); public SO-101 datasets use the default; the policy closes the loop through the cameras, so the
steady sag under gravity is compensated by what it sees. P = 32 halves the sag (M2) but the servos jitter and
heat more, and the pipeline would need re-testing.

**Options:** 16 (chosen); 32 (M2's value).

## 2026-11-09: Policy cameras: ZED left eye (cropped) and the wrist camera

**Decided:** both cameras: the ZED's left eye at FHD, cropped to x 240-1920 and resized to 840 x 540, and the
wrist camera captured at 1280 x 720 and resized to 640 x 360; 30 fps (PROTOCOL section 7,
`protocol/recording.yaml`).

**Why:** the ZED sees the whole arm, the tape and the bowl (where to go); the wrist camera sees the grasp up
close (the public SO-101 datasets and ACT/SmolVLA recipes use a scene and a wrist view). The left eye keeps
the whole folded arm in view, where the right eye cuts it off, and the calibration tools use the left eye.

**Image sizes:** ACT runs its ResNet on the stored images (SmolVLA resizes to 512 x 512 anyway), so pixels cost
training time. Measured on this laptop (ACT, batch 8, bf16): ZED 840 x 540 + wrist 1280 x 720 = 0.53 s/step,
10.0 GB; ZED 840 x 540 + wrist 640 x 360 = 0.25 s/step, 4.5 GB (the public 640 x 480 pair: 0.22 s, 4.0 GB); the
uncropped-size ZED (1680 x 1080) does not fit in 16 GB. At 840 x 540 the 5 cm box spans 52-64 px in the ZED view.
The wrist camera is resized in software, not switched to its own 640 x 360 mode, so its field of view stays the
one calibrated in M2.

**Options:** the ZED alone; the wrist camera alone; both (chosen). Right eye instead of the left (rejected).
Wrist at 1280 x 720 (2x slower training) or 640 x 360 (chosen).

## 2026-11-09: Keep the ZED's automatic exposure

**Decided:** the ZED runs with its own automatic exposure and gain during recording and evaluation; only the
V4L2 controls it exposes are set (power line frequency; gain and white balance left at their defaults). The
wrist camera's exposure and white balance can be locked through V4L2.

**Why:** the light box keeps the lighting constant (table brightness 129-157 of 255 across the view after the
final setup, 0.03 % saturated pixels), so the automatic exposure settles at similar values in every session.
Locking it needs the ZED's vendor extension-unit commands (as in Stereolabs' zed-open-capture): the UVC
controls have no exposure-auto switch. The user chose to start without that work.

**Risk, watched:** the exposure can drift when the arm or a hand fills more of the view. If S3 or S5 videos
show brightness changes that matter, port zed-open-capture's AEC/AGC control (option a below).

**Options:** (a) port the extension-unit exposure control to Python (~150 lines; exposure really fixed);
(b) keep the automatic exposure (chosen); (c) install the ZED SDK for camera access (heavy; rewrites capture).

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

## 2026-11-09: The object is a blue ring box

**Decided:** the task object is a blue ring box (a rounded cube, lid up) instead of the tape roll; PROTOCOL
amendment 2. Region, bowl, cameras and start pose are unchanged; the yaw of the initial states is used again.

**Why:** the round tape roll was hard to grasp with the leader arm (the user's experience in the dry runs).
The box is grasped on two flat faces; with the box all five dry-run episodes ended with the box in the bowl.
Blue on the black mat is easy to see in both cameras.

**Options:** keep the tape roll; the ring box (chosen).

## 2026-11-09: Policy runs start from the clipped rest pose, with a 4 deg step cap

**Decided:** (1) on the arm, a policy starts from the demonstrations' folded start pose clipped to SafeArm's
limits (shoulder lift -95.0 instead of about -103.7 deg, elbow 91.8 instead of 94.6, gripper 4.5 instead of
1.0 %); demonstrations keep starting from the folded pose itself. (Corrected 2026-11-10, PROTOCOL amendment
3: the first version used so101-core's stored rest pose, whose wrist roll is 100 deg off the demonstrations'.) (2) SafeArm's step cap for policy runs is 4 deg per 30 Hz tick
(120 deg/s), set in `protocol/recording.yaml` (`rollout.max_step_deg`).

**Why:** (1) the policy is closed loop and sees where the arm is; a start a few degrees off is the kind of
variation it should handle, and the slow replay of a demonstration from the clipped pose reproduced the
recorded states. If the first runs show trouble at the start, the alternative is a rest-zone exception in
so101-core's safety layer. (2) The demonstrations reach ~110 deg/s on the shoulder and the elbow; the default
cap (2 deg per tick, 60 deg/s) would slow the arm below what the policy plans, so the camera views would drift
from the training data and the success rate would measure the cap rather than the policy. 4 deg still stops a
sudden jump from a bad action; the limit margin and the stall check are unchanged.

**Options:** (1) accept the offset (chosen); a rest-zone exception in the safety layer; a start pose inside the
limits for demonstrations too (hard to reproduce by hand with the leader). (2) 2 deg (60 deg/s); 4 deg
(chosen); 10 deg (LeRobot record's own cap; hardly a limit).

## 2026-11-09: The object is a roll of tape (task: put the tape in the bowl)

**Decided:** the task object changes from a 3 cm cube to a roll of tape (outer diameter 37 mm, lying flat), put
into a grey ceramic pet bowl on a foot (rim 115 mm across, 73 mm high). Recorded as PROTOCOL amendment 1,
before any demonstration. Placement: the tape's region is x 190-330 mm, y -110 to +40 mm in the base frame;
the bowl's centre is at (250, 170) mm; the ZED crop is x 240-1920 at full height, resized to 840 x 540.

**Why:** the user's choice of object. The rest of the task decision below still holds (one object, a fixed
target, unambiguous success). The roll is symmetric, so its rotation needs no placing (faster resets).
Placement from the arm's model and the ZED view (captures, not committed): the bowl moved 5 cm closer to the
arm (it was 34 cm out, at the edge of reaching over a 7 cm rim); the region starts at x 190 mm because the
gripper tip rests at x 156 mm, 14 mm above the table, and ends at x 330 mm, the limit for a gripper pointing
down; the crop keeps the folded arm (to x ~260 px) and the height the gripper reaches above the bowl.

**Options:** keep the cube (S1's choice); the tape roll (chosen).

## 2026-11-08: The task is one cube into a fixed bowl

*Superseded in its object and region by the 2026-11-09 entry above; the reasoning still applies.*

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
