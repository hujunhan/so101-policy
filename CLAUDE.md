# Working notes for Claude

M3 of the SO-101 series. Plan, status and results: README.md. Decisions: DECISIONS.md.
The series plan (M0-M5) is in the so101-core README. M2 (so101-calib) is done: its README ("Handed to M3 and M5"),
docs/DESIGN.md and docs/THEORY.md say what this repo starts from.

## Current state
- 2026-11-07: repository set up (skeleton, pinned so101-core and so101-calib, smoke tests). Nothing recorded yet.
  Environment checked: LeRobot 0.6.2 (pinned via so101-core), torch 2.11 + CUDA 12.8 on the RTX 3080 Laptop.
- The user is rebuilding the workspace (light box, backdrop; materials still to buy). Data collection and
  evaluation wait for it; task choice, the protocol, the recording pipeline and training on a public dataset do
  not.
- 2026-11-08: plan agreed; GitHub repo created (public). S1 done: task = one cube into a fixed bowl; protocol
  in docs/PROTOCOL.md (30 trials, seeded list protocol/eval_states.csv, Wilson + exact McNemar). Its section 7
  (region, bowl, start pose, cameras, gain, time limit) is filled in once the workspace exists and S2 dry runs
  are done.
- 2026-11-08: S2 pipeline working (`so101-record`, `so101-check-dataset`, `so101-replay`; dry-run dataset in
  git-ignored `data/local__so101_s2_dry`). Run output is tee'd to git-ignored `logs/`: read the newest log there
  instead of asking the user to paste. Light box materials (table, foam board, Neewer GC10B, black desk mat)
  being bought.
- 2026-11-08: ZED re-location written in so101-calib (`so101-handeye relocate`, run from the so101-calib
  directory; pin bumped). Placement overlay deferred until S5 (the user to choose overlay vs template).
- 2026-11-08: S4 dry run on lerobot/svla_so101_pickplace done (README section; checkpoints in git-ignored
  outputs/train/s4_public_act). Training recipe proposed to the user (batch 8, bf16, 20-30k steps, eval_split
  0.1): not yet confirmed or recorded in DECISIONS.
- 2026-11-08: SmolVLA dry run done (README; outputs/train/s6_public_smolvla): ~3 h, matches ACT offline, 190 ms per
  new chunk. `so101-rollout` written; shadow-tested with ACT. Open for the user: start pose vs safety limits,
  rollout step cap (recommended 4 deg/tick).
- 2026-11-08: `so101-eval run|report` written (the protocol's trial runner: resume, voids repeated, trials.csv in
  results/, Wilson + McNemar); unit-tested, not yet run on the arm. Reads eval.time_limit_s / start_pose_deg /
  region from recording.yaml once PROTOCOL section 7 is filled.
- 2026-11-09: light box final (black mat taped, left side screened, lighting even). Left eye kept; ZED auto
  exposure kept (DECISIONS). ZED located: so101-calib results/relocate/20261109_214800.yaml (table at -2.4 mm =
  URDF base bottom; board 5 mm on a 1 mm mat). ZED and arm base must not move from now on.
- 2026-11-09: task changed to a tape roll (37 mm) into a grey bowl (PROTOCOL amendment 1, DECISIONS). Section 7
  filled: region x 190-330, y -110..40 mm; bowl at (250, 170) mm; ZED crop x 240-1920 -> 840 x 540.
- 2026-11-09: start pose = clipped rest pose (accepted), rollout step cap 4 deg/tick (DECISIONS, recording.yaml).
- 2026-11-09: servo P = 16 decided. Dry runs in the box: dataset local/so101_box_dry (data/, git-ignored).
- 2026-11-09: object changed to a blue ring box (PROTOCOL amendment 2). Dry runs local/so101_box_dry2: 5/5 into
  the bowl; time limit 70 s. Section 7 complete except the box's size.
- **Next:** S3 (50-100 demonstrations).

## How we work
- Claude writes and runs most of the code. The user does physical work on the arm, makes the key
  decisions and learns the concepts. Present options with a recommendation and let the user decide
  design choices, then record them in DECISIONS.md (date, decided, why, options).
- The user must be able to defend every choice, so explain the why in plain terms, briefly.
- Talk to the user in Chinese. Code, comments, commits and repo docs are in English.
- Small steps; one session ≈ 1–2 hours of the user's time. Keep physical steps few and simple (in M2 the user
  stopped a method that needed too many hand touches, and asked for fewer views).
- When a recommendation rests on something uncertain (hardware settings, library defaults, known pitfalls),
  search the web first, summarize what others report, then recommend.
- The user's terminal shows text only: for a figure, give its absolute path and `xdg-open <path>`, and say in
  words what it shows. Never "as the figure shows".
- Check claims against a baseline before writing them down; report results that go against the plan as they are.

## Rules
- The repo is public. Never commit personal information or device serial numbers. Camera images and videos can
  show people and the room: keep raw captures in the git-ignored `captures/` / `videos/`, commit only reviewed,
  cropped media. Delete images Claude viewed that show the user or the room once they are not needed.
- Real-arm motion only through so101-core's `SafeArm` (limit margin, step cap, stall detection; `move_to`,
  `follow`), or LeRobot's teleoperation/record loop with the user at the leader arm. Tell the user before
  anything moves the arm; they must be present with power in reach.
- Dependencies are pinned git commits (so101-core, so101-calib). If a fix belongs there, make it there, push,
  then bump the pin here.
- Tooling: uv, Python 3.12, ruff, pytest. Run `uv run ruff check . && uv run pytest` before committing.
- Commit and push when a step is done and verified. After a session milestone, update the status line in
  README.md and the project row in the user's twin repo (`~/code/digital-twin/projects/README.md`; that file
  may hold the user's own uncommitted edits: change only the so101 row and leave committing to the user).

## Practical notes from M1 and M2
- Device config: `SO101_CONFIG_DIR=~/code/mine/so101-core/config` (ports, rest pose, joint map with the elbow
  offset -0.127077 rad).
- Powered runs: the user runs the script in their own terminal (it asks for `yes`, and for Enter before
  releasing torque; Claude's shell cannot answer prompts) and pastes the output; Claude analyzes the logs.
  Torque-off reads (`SafeArm(...).connect(torque=False)`) Claude can run directly; they make the arm limp.
- If the follower reports "Missing motor IDs", the servo power supply is off. A single "Failed to disable
  torque ... timed out" was a one-off bus glitch (status registers were clean); retry.
- Always `uv sync --extra hw`: a plain `uv sync` removes LeRobot and with it opencv-python-headless, which
  deletes files that opencv-python shares (`cv2.aruco` disappears). Fix: `uv sync --extra hw --reinstall-package
  opencv-python`. The same reinstall fixes "The function is not implemented" from `cv2.imshow`/`waitKey`.
- ZED 2 is `/dev/video6` as a UVC device: 2560x720 side by side (HD), take the left half. Its factory file is in
  so101-calib's git-ignored `captures/zed/` (never commit or print the serial). FHD per eye is planned for M3.
- The ZED exposes for the room: a bright iPad saturates (use ~50 % brightness); overhead lights reflect in glass.
- The arm's rest pose is outside the safety limits (shoulder -102 deg vs -95); moving to rest the shoulder
  wobbles ~0.1 deg near zero gravity load. `so101-park` in so101-calib has a standing park pose.
- Rendering Mermaid diagrams: `npx -y @mermaid-js/mermaid-cli@11.4.2 -p <puppeteer.json>` with
  `{"args":["--no-sandbox"]}`; Chromium's sandbox cannot start on this Ubuntu.
