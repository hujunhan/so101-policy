# so101-policy: theory

The theory behind M3: what the learned policies are, why they work and fail the way they do, and how their
success is measured. Read it with the code. [DESIGN.md](DESIGN.md) explains how the code is built, the
[README](../README.md) has the numbers, [PROTOCOL.md](PROTOCOL.md) the evaluation rules, and
[DECISIONS.md](../DECISIONS.md) when and why each decision was made.

Notation: $o_t$ is the observation at tick $t$ (two camera images and the joint state $s_t$), $a_t$ the action
(six target positions: five joints in LeRobot degrees, the gripper in %). The control rate is 30 Hz, so one
tick is 33 ms. $\pi$ is a policy, $\pi^*$ the demonstrator. $k$ is the chunk length (how many future actions
one prediction holds).

Contents:

1. [Learning a policy from demonstrations](#1-learning-a-policy-from-demonstrations)
2. [Why behaviour cloning drifts: compounding errors](#2-why-behaviour-cloning-drifts-compounding-errors)
3. [ACT: action chunking with a transformer](#3-act-action-chunking-with-a-transformer)
4. [Several right answers: multimodal demonstrations](#4-several-right-answers-multimodal-demonstrations)
5. [SmolVLA: a pretrained vision-language model with a flow-matching action expert](#5-smolvla-a-pretrained-vision-language-model-with-a-flow-matching-action-expert)
6. [What the demonstrations must contain](#6-what-the-demonstrations-must-contain)
7. [Reading training curves and offline errors](#7-reading-training-curves-and-offline-errors)
8. [Deployment shift: when the arm's world differs from the data's](#8-deployment-shift-when-the-arms-world-differs-from-the-datas)
9. [Running a policy on position servos](#9-running-a-policy-on-position-servos)
10. [Does a learned policy need calibration?](#10-does-a-learned-policy-need-calibration)
11. [A success rate from 30 trials: the Wilson interval](#11-a-success-rate-from-30-trials-the-wilson-interval)
12. [Comparing two policies: paired trials and McNemar's test](#12-comparing-two-policies-paired-trials-and-mcnemars-test)
13. [Why the protocol is fixed in advance](#13-why-the-protocol-is-fixed-in-advance)
14. [References](#14-references)

---

## 1. Learning a policy from demonstrations

A **policy** maps what the robot sees to what it does: $a_t = \pi(o_t)$. Instead of writing it by hand
(detect the box, compute its pose, plan a grasp: M5's way), **imitation learning** fits it to examples. The
user teleoperates the follower with the leader arm; each tick LeRobot stores the observation and the leader's
position, which is the action the demonstrator chose. Fifty demonstrations of ~25 s at 30 Hz are ~37 000
pairs $(o_t, a_t)$.

**Behaviour cloning** treats this as supervised learning: find the network parameters $\theta$ that make
$\pi_\theta(o_t)$ close to $a_t$ over the dataset,

$$\min_\theta \ \mathbb{E}_{(o, a) \sim \mathcal{D}} \big[\, \ell(\pi_\theta(o), a) \,\big],$$

with $\ell$ an L1 or L2 distance (ACT) or a generative loss (SmolVLA, section 5). No reward, no simulator, no
model of the arm or the box: everything the policy knows about the task comes from the pairs.

What the policy receives:

- **Images**: the ZED's left eye (840 x 540, where the box and the bowl are) and the wrist camera (640 x 360,
  the grasp up close). Pixel values only; the policy is not told where either camera is.
- **Joint state**: the follower's six readings in LeRobot units. It tells the policy where the arm is, which
  the images show only roughly.
- (SmolVLA only) **a task sentence**: "Put the blue box in the bowl."

## 2. Why behaviour cloning drifts: compounding errors

Supervised learning assumes the test inputs come from the training distribution. A policy breaks that
assumption by acting: its own small error moves the arm to a state slightly unlike any in the demonstrations,
where its error is larger, which moves it further away. Ross and Bagnell showed that if the policy errs with
probability $\epsilon$ per step on the demonstrator's states, its expected cost over a horizon of $T$ steps
can grow as $O(T^2 \epsilon)$, against $O(T \epsilon)$ for a learner that is trained on its own states
(DAgger). This is **covariate shift**, and it is the central weakness of behaviour cloning.

Three practical answers, all used or available here:

- **Shorter effective horizons**: predict and commit to several actions at once (section 3). Decisions are
  made $T/k$ times instead of $T$.
- **Data that cover the recoveries**: demonstrations that start from varied poses and positions, and that
  correct small mistakes, show the policy what to do near (not only on) the ideal path.
- **Closed loop**: the policy looks again; a state slightly off the path still looks like "the box is there,
  the gripper is here", and the right action from there is often still in the data.

## 3. ACT: action chunking with a transformer

ACT (Zhao et al. 2023) was designed for exactly this setting: low-cost arms, 50 demonstrations, fine
manipulation. LeRobot's implementation, with its defaults, is what M3 trains.

**From images to tokens.** Each camera image goes through a ResNet18 (pretrained on ImageNet, fine-tuned with
the rest). Its last feature map has one 512-channel vector per 32 x 32 patch: for the ZED's 840 x 540 about
27 x 17 = 459 vectors, for the wrist's 640 x 360 about 20 x 12 = 240. Each vector becomes a token, with a 2-D
sinusoidal position code so the transformer knows where in the image it came from. The joint state and a
latent variable $z$ (below) become two more tokens.

**Transformer encoder-decoder.** A 4-layer transformer encoder mixes all tokens (every patch can attend to
every other, across both cameras and the state). A decoder with $k = 100$ learned queries attends to them and
outputs 100 actions at once: the next 3.3 s of target positions. The design follows DETR (Carion et al.
2020): a fixed set of queries, each producing one output.

**Action chunking.** Predicting a chunk instead of one action has two effects. The horizon of decisions
shrinks by $k$ (section 2). And a chunk is a coherent motion: a demonstrator who pauses or hesitates
produces actions that a single-step model sees as noise, while a chunk captures the whole movement around the
pause.

**Executing the chunk.** LeRobot's ACT, with `n_action_steps` = 100, runs the network once, then executes the
100 actions open loop, one per tick, before it looks again. Between those moments the policy is blind to what
happens (the box slipping, a missed grasp). The ACT paper's alternative is **temporal ensembling**: query every
tick and average all chunks that predict the current step, weighted $w_i = e^{-m i}$ with $i = 0$ the oldest
($m = 0.01$ in the paper). It is smoother and reacts every tick, at 30x the network calls (ACT's 32 ms per call
fits in a 33 ms tick only just). A middle way is executing fewer actions per chunk (e.g. 25). The pilot uses
the default; this is a knob to try if failures look like "it did not notice".

**Normalization.** States and actions are standardized with the dataset's mean and standard deviation per
joint, $(x - \mu)/\sigma$, so all joints and the gripper matter equally in the loss; images with ImageNet's
statistics, as the ResNet expects. The predicted actions are un-standardized before they go to the arm.

**Loss.** The L1 distance between the predicted and the demonstrated chunk, plus the CVAE term (section 4):

$$\mathcal{L} = \sum_{i=0}^{k-1} \big| \hat a_{t+i} - a_{t+i} \big| \;+\; \beta\, D_\text{KL}\big(q(z \mid a_{t:t+k}, s_t)\ \|\ \mathcal{N}(0, I)\big), \qquad \beta = 10 .$$

L1 rather than L2: it is less pulled by the occasional large error (a glitch, a different choice), and the
ACT authors found it gave more precise actions.

## 4. Several right answers: multimodal demonstrations

A demonstrator does not always do the same thing in the same situation. The box can be grasped from two
opposite faces; the arm can go around the bowl on the left or the right; the demonstrator waits 0.5 s one
time and 2 s the next. The data then contain several right answers for one observation: the action
distribution is **multimodal**.

A regression loss averages them. Half the demonstrations grasp at a yaw of 0 deg and half at 90 deg; the mean,
45 deg, grasps a corner and fails. Half wait, half go; the mean starts slowly and late. This is not a lack of
data, it is the wrong model for the data.

**ACT's answer: a conditional VAE.** During training, an extra transformer encoder sees the true action chunk
(and the state) and compresses it into a 32-dimensional latent $z$, drawn from $q(z \mid a, s)$. The decoder
is given $z$ along with the images. So $z$ can carry "which of the right answers this demonstration took",
and the decoder no longer has to average. The KL term keeps $q$ close to a standard normal, so that the
decoder learns to work with $z$ values a normal distribution produces. At deployment there is no true chunk;
ACT sets $z = 0$, the prior's mean, and gets one consistent choice. (This is also why the held-out loss logged
in training, computed with $z = 0$, is not comparable with the training loss, which uses $z$ from the true
actions: README, S4 dry run.)

**Generative policies** model the whole distribution of actions and sample from it: diffusion policies (Chi
et al. 2023) and flow matching (SmolVLA, pi0). Each sample is one coherent choice, never an average of two.

**The cheap answer is consistent demonstrations.** One demonstrator, one strategy: start soon, approach the
same way, grasp the faces nearest the box's yaw. Consistency costs nothing at recording time and removes
modes the model would otherwise have to separate.

## 5. SmolVLA: a pretrained vision-language model with a flow-matching action expert

A **vision-language-action model** (VLA) starts from a model pretrained on images and text, and adds a part
that outputs robot actions. The bet is that what it already knows about images (objects, colours, positions)
transfers, so fewer demonstrations are needed and the policy is more robust to visual change.

SmolVLA (Shukor et al. 2025), 450 M parameters:

- **The vision-language model**: SmolVLM2-500M. Each image is resized with padding to 512 x 512 and encoded
  into 64 visual tokens; the task sentence into text tokens; the joint state is projected into one token.
  Only the first 16 of its layers are used (later layers add cost and help little for control). It is
  **frozen** in our fine-tuning.
- **The action expert**: a smaller transformer (0.75 x the VLM's width, ~100 M parameters, the only part
  trained here) that produces a chunk of 50 actions. Its layers alternate cross-attention to the VLM's
  features and self-attention among the action tokens.
- Pretrained on community datasets recorded with SO-100 arms, which is why it is a natural start for this arm.

**Flow matching.** The expert does not regress the actions; it learns to turn noise into an action chunk.
Draw a noise chunk $\varepsilon \sim \mathcal{N}(0, I)$ and a time $\tau \in [0, 1]$, and mix
(LeRobot's convention):

$$x_\tau = \tau\, \varepsilon + (1 - \tau)\, A, \qquad A = a_{t:t+50} .$$

Along this straight line from the data ($\tau = 0$) to the noise ($\tau = 1$) the velocity is constant,
$u = \varepsilon - A$. The network $v_\theta(x_\tau, \tau, o_t)$ is trained to predict it:

$$\mathcal{L} = \mathbb{E}_{A,\, \varepsilon,\, \tau}\, \big\| v_\theta(x_\tau, \tau, o_t) - (\varepsilon - A) \big\|^2 .$$

To act, start from fresh noise at $\tau = 1$ and integrate back to $\tau = 0$ with 10 Euler steps,
$x \leftarrow x - 0.1\, v_\theta(x, \tau, o_t)$. Different noise gives different, individually coherent
chunks: the multimodality of section 4 is handled by sampling, not averaging. The cost is 10 passes of the
expert per chunk: 190 ms on this laptop, against 32 ms for ACT.

**Asynchronous inference.** At 30 Hz, a 190 ms computation stalls the arm for about 6 ticks every 50 if the
loop waits for it. SmolVLA's authors run inference in parallel: when part of the current chunk remains, the
next chunk is computed from a fresh observation and spliced in when ready. Fewer flow-matching steps (e.g. 5)
halve the time at some cost in quality.

## 6. What the demonstrations must contain

The policy can only interpolate between demonstrations; it cannot invent what it never saw.

- **Coverage of the initial states.** The evaluation draws box positions uniformly over the region and yaws
  over [0, 90) deg. If the demonstrations leave a corner of that space empty, the trials there test
  extrapolation. The first 20 demonstrations covered the near third of the region once; the pilot's misses
  and its poor wrist roll (worse than holding still offline) are what that predicts. Positions in the
  demonstrations are drawn independently of the evaluation list, so the list is never trained on.
- **Consistency**: section 4.
- **No random waits.** A demonstrator who waits 2 s after "recording" teaches a wait whose length the images
  cannot predict. The S4 dry run on a public dataset showed the policy starting ~1 s before a demonstrator who
  had waited 2 s: the error was timing, not motion. Start soon, stop soon after returning.
- **Recoveries.** A demonstration that regrasps after a near miss shows the policy what to do after its own
  near miss (section 2).
- **Action is the leader, state is the follower.** The follower lags the leader by a few frames (servo lag,
  ~0.1 s) and sags under gravity. The policy learns to command what the human commanded, including the
  over-command that makes the follower arrive; it does not learn to copy the follower's positions.
- **How many.** Community reports for ACT on single pick-and-place tasks on SO-100/101 put 50 demonstrations
  at roughly 70-90 % in distribution; in the pilot the held-out error stopped improving after 5k steps. More
  demonstrations help where coverage is thin, not uniformly.

## 7. Reading training curves and offline errors

**Training loss** falls for as long as training runs: the network memorizes the demonstrations. **Held-out
loss** (episodes kept out by `eval_split`) tells whether that carries over to new episodes. In the S4 dry run
it was flat from 2k steps while training loss kept falling; in the pilot the held-out action error stopped
improving after 5k steps. With tens of demonstrations, most of a long training run is memorization.

**Open-loop action error** (`so101-offline-eval`): feed recorded observations, compare the predicted chunk
with what the demonstrator did, per joint and horizon. The **"hold still" baseline** (predict the current
state for every future action) is what a policy that learned nothing would score; a useful policy is clearly
below it at longer horizons.

What offline numbers cannot do:

- **They do not measure success.** On the arm the policy's actions change its next observation. An error early
  in a chunk can be corrected next time it looks (a 10 deg error in a reach the camera then fixes), or can
  compound (a grasp a few mm off that misses). Offline error weights every action equally; success depends on
  a few critical ones.
- **They penalize a different but valid choice.** If the demonstrator grasped the left face and the policy
  plans the right face, the offline error is large and the trial on the arm may still succeed.
- **They cannot rank close policies.** ACT and SmolVLA differed by less than the noise of 5 held-out episodes
  (README). That is what the 30 trials are for.

So offline evaluation is a **check of the pipeline and of training** (units, cameras, normalization; is it
learning at all?), and the checkpoint is chosen by rule (the final one), not by offline score.

## 8. Deployment shift: when the arm's world differs from the data's

A policy trained from pixels learns whatever in the pixels predicts the demonstrator's actions, including
things nobody meant it to use: where the back wall's edge is, how bright the table is, what is visible at the
left. If any of these change between recording and evaluation, the inputs leave the training distribution
and the policy's errors grow, without any error message. The protocol fixes the scene for this reason.

The kinds of shift that matter here:

| Shift | What changes in the inputs | Measured / handled |
|---|---|---|
| **Camera pose** | every pixel moves: the box appears at another place, and so does the background the policy may use as a reference | ZED image 9.3 px (of 540, ~1 deg) lower on 2026-11-10 than in the demonstrations; zip-tie mount to be replaced (DESIGN section 9) |
| **Lighting, exposure** | brightness and colour of everything | light box, one light at fixed settings; ZED automatic exposure (M4's subject) |
| **Backdrop** | the background tokens | left side screened; people and animals out of view |
| **Start pose** | the first joint state and wrist view | amendment 3: start where the demonstrations start |
| **Speed** | the images seen along the way (a slower arm sees more frames per motion) | step cap above the demonstrations' peak speed |
| **Servo gain** | how far the arm sags for a given command, so the state that follows an action | P = 16 in both |

How large a camera shift is too large is not well known. Studies with large perturbations find steep drops (a
surgical-robot study: ACT from 95 % to 35 % for ±3 cm and to 15 % for ±10 deg of camera motion); a study of
camera conditioning found that policies trained without camera information rely on static background cues to
infer where the camera is. For ~1 deg there is no clean number; the pilot succeeded with the camera 4.4 px off,
so small shifts do not break it, but they add an uncontrolled factor to every trial. (Sources: section 14.)

**Data augmentation** makes the policy indifferent to small shifts by showing them during training: random
affine transforms (rotation, translation) and colour jitter applied to the training images. LeRobot has both
(±5 deg, ±5 % translation; brightness, contrast, saturation, hue, sharpness), off by default. A ±5 %
translation of a 540-row image is ±27 px, enough to cover the drift seen so far. The cost is slower learning
(the network sees a harder problem) and a slightly blurrier sense of absolute position.

**The wrist camera is immune to scene-camera motion**: it moves with the gripper, and it sees the box at the
moment that matters most, the grasp. This is one reason a scene camera plus a wrist camera is the common
recipe for these arms.

## 9. Running a policy on position servos

The servos are position controlled: each tick the policy sends a target and the servo's internal loop drives
toward it (M2: a spring with stiffness set by P, a friction band, ~50-70 ms to respond). The policy's actions
are therefore **targets, not trajectories**: like the demonstrator's leader positions, they lead the arm.

**The step cap and the lag.** `SafeArm` caps each target at 4 deg from the last command (120 deg/s at 30 Hz),
above the demonstrations' ~110 deg/s peak, so it does not slow a policy that moves like the demonstrator.
A joint moving at speed $\dot q$ reads behind its command by roughly $\dot q \times t_\text{lag}$: at 120 deg/s
and ~0.1 s that is 12 deg, and 12-16 deg was measured in the pilot runs. A stall watch must not confuse this
lag with a blocked joint; hence the threshold of 25 deg for 0.5 s (a blocked joint keeps a growing gap, a
moving one does not).

**Timing.** At 30 Hz a tick is 33 ms. ACT's network (32 ms) runs once per 100 ticks; between, a tick costs
6-8 ms. SmolVLA's 190 ms per chunk does not fit in a tick (section 5). The camera frames used each tick are
the newest available; their age is logged, and a frame older than 0.5 s stops the run, because a policy
acting on an old image acts on a world that has moved.

**Why the policy tolerates sag.** At P = 16 the shoulder and elbow sag by up to ~3 deg under gravity (M2).
The demonstrations were recorded with the same sag: the human commanded past the target until the follower
got there, and the policy learned those commands. And it closes the loop through the cameras: if the gripper
is low, it sees it. M2's feed-forward compensation is not needed here, and would change the arm's response
from the one the data were recorded with.

## 10. Does a learned policy need calibration?

The policy's inputs are raw pixels and LeRobot joint readings; its outputs are LeRobot joint targets. None of
M2's results (camera intrinsics, hand-eye transforms, the elbow's zero offset) appear in that chain. The
rollout converts LeRobot degrees to radians and back with the same joint map, so the offset cancels; it only
shifts `SafeArm`'s limits. The policy is calibrated implicitly: the demonstrations relate what the cameras
see to which joint targets work, in whatever frame the cameras and joints happen to be in.

The price of implicit calibration is that **it holds only for the setup it was learned in**. An explicit
calibration can be redone after a camera moves (`relocate`, a few minutes) and a classical pipeline (M5) is
then right again; a learned policy has to be retrained, or trained with augmentation for robustness to the
change. That is the trade M5 measures: engineering effort to calibrate against effort to collect data and
sensitivity to change.

Where calibration is used in M3: the evaluation region and the bowl are defined in the base frame (millimetres
on the table), placed using the ZED pose from `relocate`; the joint map keeps the safety limits right; the
analysis of demonstrations and failures by box position converts ZED pixels to the table.

## 11. A success rate from 30 trials: the Wilson interval

Each trial succeeds or fails. If the policy's true success probability is $p$ and trials are independent,
the number of successes in $n$ trials is binomial, and the observed rate $\hat p = x/n$ is an estimate whose
spread is $\sqrt{p(1-p)/n}$: about 7 points at $p$ = 0.8 and $n$ = 30.

The textbook (Wald) interval $\hat p \pm z\sqrt{\hat p(1 - \hat p)/n}$ is poor at small $n$ and near 0 or 1:
at 30/30 it gives [1, 1], certainty from 30 trials. Its actual coverage at $n$ = 30 can fall well below 95 %
(Brown, Cai and DasGupta 2001). The **Wilson interval** inverts the score test instead, and stays honest:

$$\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \ \pm\ \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat p(1 - \hat p)}{n} + \frac{z^2}{4n^2}}, \qquad z = 1.96 .$$

| Successes of 30 | $\hat p$ | Wald 95 % | Wilson 95 % |
|---:|---:|---|---|
| 24 | 0.80 | 0.66-0.94 | 0.63-0.90 |
| 30 | 1.00 | 1.00-1.00 | 0.89-1.00 |

The centre is pulled toward 0.5 and the interval is asymmetric near the ends, as a bounded quantity should
be. Its width is the honest resolution of 30 trials: about ±14 points at 80 %. Differences smaller than that
are reported as such (PROTOCOL section 4).

## 12. Comparing two policies: paired trials and McNemar's test

ACT and SmolVLA run **the same 30 initial states in the same order**. Some states are easy (box near the
middle, square to the gripper) and some hard; with the same states, that difficulty is the same for both, and
the comparison looks only at what differs. Each trial falls in one of four cells:

| | SmolVLA succeeds | SmolVLA fails |
|---|---:|---:|
| **ACT succeeds** | both (no information about which is better) | $b$ |
| **ACT fails** | $c$ | neither (no information) |

If the two policies were equally good, each **discordant** trial ($b + c$ of them) would be equally likely to
go either way. **McNemar's exact test** is the binomial test of that: under the null hypothesis
$b \sim \text{Binomial}(b + c, \tfrac12)$, and the two-sided p-value is

$$p = \min\Big(1,\ 2 \sum_{i=0}^{\min(b, c)} \binom{b + c}{i} 2^{-(b+c)}\Big).$$

Examples: $b = 8, c = 1$ gives $p = 0.039$; $b = 7, c = 1$ gives $p = 0.070$; even $b = 5, c = 0$ gives only
$p = 0.0625$: five trials are too few to tell, however lopsided. The concordant trials do not enter, whereas
an unpaired test comparing two rates would treat the shared easy and hard states as noise in both. The paired
design is what lets 30 trials show a difference at all; even so, only clearly lopsided splits of enough
discordant trials reach significance, so both counts and the p-value are reported, not a verdict.

## 13. Why the protocol is fixed in advance

Every analysis has choices: which checkpoint, how many trials, what counts as success, when a trial is void,
which positions. Made after seeing results, each choice can be made, without intent, in the direction that
flatters the result (the "garden of forking paths", Gelman and Loken). The protocol was written and committed
before the first demonstration, with the open values (region, time limit) fixed by rule before any policy ran,
and later changes made only as dated amendments with reasons. In particular:

- **The final checkpoint**: choosing the checkpoint that does best on the evaluation trials turns the test set
  into a validation set and overstates success.
- **No retries, voids only for faults outside the policy**: otherwise bad trials can be explained away.
- **The first thing that went wrong** is the failure label: one label per failure, decided from video, so the
  taxonomy counts causes, not symptoms.
- **The time limit by rule** (3 x the median dry-run duration): not tuned to how the policies behave.

The same rules let M4 and M5 compare against M3 without re-arguing them.

## 14. References

- T. Z. Zhao, V. Kumar, S. Levine, C. Finn, "Learning fine-grained bimanual manipulation with low-cost
  hardware", RSS 2023 (ACT, action chunking, temporal ensembling).
- M. Shukor et al., "SmolVLA: a vision-language-action model for affordable and efficient robotics",
  arXiv:2506.01844, 2025 (SmolVLA, asynchronous inference).
- K. Black et al., "pi0: a vision-language-action flow model for general robot control", arXiv:2410.24164,
  2024 (flow-matching action expert).
- Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le, "Flow matching for generative modeling", ICLR 2023;
  X. Liu, C. Gong, Q. Liu, "Flow straight and fast: learning to generate and transfer data with rectified
  flow", ICLR 2023.
- C. Chi et al., "Diffusion policy: visuomotor policy learning via action diffusion", RSS 2023.
- K. Sohn, H. Lee, X. Yan, "Learning structured output representation using deep conditional generative
  models", NeurIPS 2015 (CVAE); D. P. Kingma, M. Welling, "Auto-encoding variational Bayes", ICLR 2014.
- N. Carion et al., "End-to-end object detection with transformers", ECCV 2020 (DETR).
- K. He, X. Zhang, S. Ren, J. Sun, "Deep residual learning for image recognition", CVPR 2016 (ResNet).
- S. Ross, J. A. Bagnell, "Efficient reductions for imitation learning", AISTATS 2010; S. Ross, G. Gordon,
  J. A. Bagnell, "A reduction of imitation learning and structured prediction to no-regret online learning",
  AISTATS 2011 (compounding errors, DAgger).
- "Overcoming imperfect kinematics in surgical robotics through sim-to-real visuomotor learning",
  arXiv:2606.21396, 2026 (ACT under camera translation and rotation; numbers from the paper's summary);
  "Do you know where your camera is? View-invariant policy learning with camera conditioning",
  arXiv:2510.02268, 2025 (background cues, extrinsics conditioning for ACT, Diffusion Policy, SmolVLA).
- W. Pumacay et al., "THE COLOSSEUM: a benchmark for evaluating generalization for robotic manipulation", RSS
  2024 (perturbation factors including camera pose).
- E. B. Wilson, "Probable inference, the law of succession, and statistical inference", JASA 22, 1927;
  L. D. Brown, T. T. Cai, A. DasGupta, "Interval estimation for a binomial proportion", Statistical Science
  16(2), 2001.
- Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages",
  Psychometrika 12, 1947; M. W. Fagerland, S. Lydersen, P. Laake, "The McNemar test for binary matched-pairs
  data: mid-p and asymptotic are better than exact conditional", BMC Medical Research Methodology 13, 2013.
- A. Gelman, E. Loken, "The garden of forking paths", 2013.
- LeRobot (pinned 0.6.2): `policies/act`, `policies/smolvla`, `datasets/transforms.py` (image augmentation).
