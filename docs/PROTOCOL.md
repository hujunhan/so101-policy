# Evaluation protocol (pre-registered)

Written 2026-11-08, before any demonstration is recorded or any policy is trained. Every real-robot
evaluation in M3 follows it, and M4 and M5 reuse it unchanged. Code: `src/so101_policy/protocol.py`.

**Changes.** Before the first policy run, the open items in section 7 may be filled in, and other changes are
allowed only as dated amendments at the end of this file, each with its reason. After the first policy run
nothing changes. A deviation during a run is reported, not hidden.

**Amendment 1 (2026-11-09, before any demonstration): the object is a roll of tape, not a cube.** Read "cube"
as "tape roll" in sections 1-5; the yaw in the initial states is not used, and the region is 14 x 15 cm.
Details at the end of this file.

**Amendment 2 (2026-11-09, before any demonstration for training): the object is a blue ring box.** It
replaces the tape roll of amendment 1; the yaw of the initial states is used again. Details at the end.

## 1. Task

Pick one cube (about 3 cm, solid matte colour) from the table and place it in a bowl at a fixed position.

- **Start**: the cube lies flat at one of the listed initial states (section 2); the bowl is at its fixed
  position; the arm is at the fixed start pose; nothing else in the workspace.
- **Success**: when the episode ends, the cube lies entirely inside the bowl, the gripper is open and not
  touching the cube, and no person touched the cube, the bowl or the arm during the trial.
- **Time limit**: three times the median duration of the dry-run demonstrations recorded in S2 (rounded up to
  the next 5 s), fixed and committed in section 7 before any policy runs. The episode ends at the time limit
  or when the operator sees that it has finished (the arm is back near the start pose, or still for 5 s).

## 2. Initial states

- **30 trials per policy and condition.**
- The cube's position is uniform over a rectangular region of roughly 20 x 15 cm on the table and its yaw is
  uniform in [0, 90) degrees (a cube looks the same after a quarter turn).
- The list is drawn once with the fixed seed 20261108 in normalized region coordinates and committed as
  `protocol/eval_states.csv` (a test checks that the file matches the seed). Every policy and every
  condition in M3, M4 and M5 runs the **same list in the same order**, so policies are compared trial by trial.
- The region's corners in the base frame, the bowl position and the start pose are fixed in section 7 once the
  workspace is rebuilt, before the first demonstration. Demonstrations cover the same region, with positions
  drawn independently of this list.
- Tolerance when placing the cube: within about 5 mm and 5 degrees of the listed state, checked against an
  overlay on the camera view (built in S2).
- Any list used for tuning (for example a validation set) uses a different seed. The evaluation list is never
  used for tuning or for choosing a checkpoint.

## 3. Running the trials

- **Checkpoint**: the final checkpoint of the training run, chosen by this rule before evaluation.
- **No retries**: each trial runs once. A trial is **void** only for a fault outside the policy (a bus error,
  a camera dropout, an emergency stop for a person entering the workspace); it is logged with its cause and
  repeated at the end of the session. Voids are reported.
- **Scene**: lighting, backdrop, camera settings and servo gain as in the demonstrations; no people or animals
  in the camera views; the ZED crop and settings are the same as in training.
- **Records**: each trial is recorded on video (all policy camera streams plus the observations and actions),
  and one row is written to the trial log.

**Trial log** (`results/<policy>/<condition>/trials.csv`, one row per trial):

| Column | Content |
|---|---|
| `trial` | 1-30, as in `eval_states.csv` (a repeated void trial keeps its number) |
| `success` | 1 or 0 |
| `failure` | empty on success, else one label from section 5 |
| `duration_s` | time to success, or the time limit |
| `void` | empty, or the cause of the void |
| `notes` | free text |

## 4. Analysis

- **Primary outcome**: success rate per policy and condition with a **95 % Wilson interval**.
- **Comparing two policies** (ACT vs SmolVLA, or one policy under two conditions in M4): the **exact McNemar
  test** on the paired trials (two-sided, alpha 0.05). Only the discordant trials count, those where one
  succeeds and the other fails. Report both counts and the p-value, not only "significant or not".
- **Secondary**: time to success (median, successful trials only); the failure distribution (section 5);
  success by position in the region (a scatter of the 30 states), described rather than tested.
- With 30 trials the interval half-width is about 14 points at an 80 % success rate. Differences smaller than
  that are reported as such, not as findings.

## 5. Failure taxonomy

Each failed trial gets exactly one label, the **first thing that went wrong**, decided from the video:

| Label | Meaning |
|---|---|
| `grasp_miss` | The gripper closed without the cube in it |
| `drop_in_transit` | The cube was grasped, then dropped before the bowl |
| `place_miss` | The cube was released but lies outside the bowl |
| `no_release` | The cube was not released, or was left stuck on the bowl rim |
| `stuck` | Freezes, oscillates, or runs out of time without one of the above |
| `collision` | Hits the bowl, the table or itself hard enough to stop or move things, or a safety stop |
| `other` | Anything else, described in the notes |

Repeated attempts count as one trial: a policy that misses the grasp, then regrasps and succeeds within the
time limit, is a success.

## 6. What is reported

For each policy and condition: n, successes, the rate with its Wilson interval, voids, median time to success,
and the failure counts. For each comparison: the discordant counts and the McNemar p-value. Plus one video
per policy with successes and typical failures.

## 7. Fixed before data (open)

To fill in once the workspace is rebuilt and S2's dry runs are done, and to commit before the first
demonstration (region, bowl, start pose, objects) and before the first policy run (time limit):

Base frame: so101-core's model frame (x forward from the arm, y to its left, z up), located in the light box by
so101-calib's `relocate` on 2026-11-09 (`results/relocate/20261109_214800.yaml`; the table is at z = -2.4 mm,
the black mat 1 mm above it). The ZED and the arm's base must not move; if either does, relocate again and
check the region in the camera view before any further trial.

- [x] **Region**: x 190-330 mm, y -110 to +40 mm (14 x 15 cm) of the tape roll's centre. u runs along x
  (u = 0 at x = 190 mm, nearest the arm), v along y (v = 0 at y = -110 mm, nearest the ZED). Why there: the
  gripper tip rests at x 156 mm, 14 mm above the table, so closer states would put the tape under it; beyond
  x 330 mm the arm cannot point its gripper down at the table; below y -110 mm the tape leaves the ZED's view.
- [x] **Bowl**: grey ceramic pet bowl on a foot, rim outer diameter 115 mm, 73 mm high; centre at x 250 mm,
  y +170 mm (300 mm from the base axis), behind the region as the ZED sees it, so it hides none of it. Its
  place is marked on the mat. **Object**: a blue ring box (a rounded cube; size to be measured), lying on its
  base, lid up (amendment 2).
- [x] **Start pose**: the arm's folded rest pose, clipped to SafeArm's joint limits (with its 5 deg margin):
  pan 8.6, lift -95.0, elbow 91.8, wrist flex 76.0, wrist roll -92.5 deg, gripper 4.5 % (LeRobot units). The
  rest pose itself has the shoulder at -102.4 and the elbow at 96.6, outside the limits, so policy runs start
  7.4 deg (shoulder) and 4.8 deg (elbow) above where demonstrations start; accepted (DECISIONS 2026-11-09).
  Policies run with SafeArm's step cap at 4 deg per tick (120 deg/s).
- [x] **Cameras**: ZED 2 left eye, FHD (1920 x 1080 per eye) over UVC, crop x 240-1920 at full height, resized to
  840 x 540; wrist camera 1280 x 720 MJPG; both 30 fps (`protocol/recording.yaml`). The ZED keeps its automatic
  exposure (DECISIONS 2026-11-09); lighting: one Ulanzi UA12 at 10 %, daylight preset.
- [x] **Servo P gain**: 16 in recording and evaluation (DECISIONS 2026-11-09).
- [x] **Time limit: 70 s.** The five dry-run demonstrations with the ring box in the light box
  (`local/so101_box_dry2`, 2026-11-09) lasted 20.3, 21.0, 23.0, 25.9 and 33.9 s; 3 x the median 23.0 s = 69 s,
  rounded up to 70 s.

## Amendments

### Amendment 1, 2026-11-09: a tape roll instead of a cube

Made before any demonstration was recorded or any policy run, when the workspace was built.

- **Object**: a roll of tape (outer diameter 37 mm), lying flat with its hole up, instead of a 3 cm cube.
  Sections 1-5 apply with "cube" read as "tape roll": success is the whole roll inside the bowl, the gripper open
  and not touching it, no person touching anything during the trial; the failure labels are unchanged.
- **Initial states**: the roll looks the same at any rotation, so the yaw column of `eval_states.csv` is not
  used (the operator places the roll's centre only, within about 5 mm). The list itself is unchanged: same
  seed, same positions in normalized coordinates, same order.
- **Region**: 14 x 15 cm instead of roughly 20 x 15 cm: the arm's reach with the gripper pointing down and the
  gripper's rest position bound it (section 7).
- **Why**: the user's choice of task object. Effect on the comparison: none between policies (all run the same
  list); results are not comparable with any cube results, of which there are none.

### Amendment 2, 2026-11-09: a blue ring box instead of the tape roll

Made after five dry-run episodes with the tape roll and five with the box (S2-style checks of the pipeline in
the light box), before any demonstration for training was recorded and before any policy run.

- **Object**: a blue ring box (a rounded cube), lying on its base with the lid up, instead of the tape roll:
  the round roll was hard to grasp with the leader. Sections 1-5 apply with "cube" read as "ring box".
- **Initial states**: the box looks the same after a quarter turn, as the cube did, so the yaw column of
  `eval_states.csv` is used again (placing tolerance about 5 mm and 5 deg). The list is unchanged.
- Region, bowl, start pose and cameras are as in section 7.

