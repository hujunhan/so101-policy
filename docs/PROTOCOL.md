# Evaluation protocol (pre-registered)

Written 2026-11-08, before any demonstration is recorded or any policy is trained. Every real-robot
evaluation in M3 follows it, and M4 and M5 reuse it unchanged. Code: `src/so101_policy/protocol.py`.

**Changes.** Before the first policy run, the open items in section 7 may be filled in, and other changes are
allowed only as dated amendments at the end of this file, each with its reason. After the first policy run
nothing changes. A deviation during a run is reported, not hidden.

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

- [ ] Region corners (x, y in mm, base frame) and which side u and v run along
- [ ] Bowl position and size; cube size and colour
- [ ] Start pose (joint angles)
- [ ] Cameras: streams, resolution, crop, frame rate, exposure and white balance settings
- [ ] Servo P gain during recording and evaluation
- [ ] Time limit (3 x median dry-run duration, rounded up to 5 s)

## Amendments

None yet.
