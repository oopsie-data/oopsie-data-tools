# AI_CONTEXT.md — oopsie-data-tools setup guide for AI assistants

This file is a step-by-step setup skill for AI tools helping a user integrate `oopsie-data-tools` into their robot codebase. Work through the sections in order, asking the user the listed questions before writing any config files.

---

## 1. Verify prerequisites

Ask the user to confirm:
- [ ] Python 3.8 or newer is installed. 3.8 is the supported floor and is tested in CI alongside 3.10 and 3.12; note 3.8 itself has been end-of-life since October 2024.
- [ ] `uv` (preferred) or `pip` is available.
- [ ] They have completed the registration form at https://forms.gle/9arwZHAvRjvbozoT7 and received a **lab ID** and **HuggingFace token**. If not, send them there first — nothing else works without these.

---

## 2. Installation

```bash
git clone https://github.com/oopsie-data/oopsie-data-tools
cd oopsie-data-tools
uv sync          # or: pip install -e .
```

Confirm the install succeeded before proceeding.

---

## 3. Where configs live

The two configs are looked up through **separate chains**, because they belong to different
things. In each chain the first location that exists wins.

**Credentials** (`contributor_config.yaml`) belong to the user and are shared by every project:

1. `$OOPSIE_CONFIG_DIR` — explicit override
2. `~/.config/oopsie-data` (or `$XDG_CONFIG_HOME/oopsie-data`)
3. the repository's `configs/` directory — only when working from a clone

**Robot profiles** belong to the robot code that loads them, and are *never* looked up in the
user config directory:

1. `$OOPSIE_ROBOT_PROFILES_DIR` — explicit override
2. `./robot_profiles` or `./configs/robot_profiles`, relative to the working directory
3. the repository's `configs/robot_profiles` — only when working from a clone

So: write the contributor config to `configs/` when working from a clone, otherwise to
`~/.config/oopsie-data/`. Put the robot profile next to the user's robot code (a
`robot_profiles/` directory beside their eval script is the normal choice) and load it by
explicit path. To keep either somewhere custom, have them persist it in their shell rc file:

```bash
echo 'export OOPSIE_CONFIG_DIR=/path/to/oopsie-config' >> ~/.bashrc          # or ~/.zshrc
echo 'export OOPSIE_ROBOT_PROFILES_DIR=/path/to/profiles' >> ~/.bashrc
```

`--config-dir <dir>` overrides the credential location for one run. It is a flag on
`oopsie-data` itself, so it goes **before** the subcommand — `oopsie-data --config-dir <dir>
upload --path ...`. Placing it after the subcommand is rejected. When unsure what is actually
in use, run `oopsie-data show-config`: it prints both chains, the location that wins, and the
lab id and token in effect.

Below, `<config-dir>` means the resolved credential location and `<profiles-dir>` the resolved
profile location.

---

## 3b. Fast path for section 4: `oopsie-data init`

If the user is at a terminal, have them run:

```bash
oopsie-data init
```

It asks which config directory to use — defaulting to the per-user one, never the checkout,
because the file holds a token and a token inside a git working tree can be committed — then
for the lab id and HuggingFace token, rejects the `your_lab_id` placeholder, and writes
`contributor_config.yaml` (mode 0600). `--lab-id`, `--hf-token`, `--no-verify-token` and
`--force` skip the corresponding prompts, so it also runs unattended.

The token check is **advisory**: a token that fails to verify is reported as a warning and
saved anyway, and `init` still exits 0. Only the lab id is actually rejected. You find out a
token is wrong at `oopsie-data upload`.

It does **not** create robot profiles. Run `oopsie-data new-profile` to write a commented
skeleton into `./robot_profiles/`, then fill it in — see section 5. The skeleton deliberately
does not load until you do, so a half-edited profile cannot stamp placeholder metadata into
recorded episodes.

---

## 4. Contributor config (`<config-dir>/contributor_config.yaml`)

Ask the user:
1. **What is your lab ID?** (exact string provided at registration — capitalization matters; a wrong value will block access to the lab-specific HuggingFace repo)
2. **What is your HuggingFace token?**

Then write/update `<config-dir>/contributor_config.yaml`:

```yaml
lab_id: <EXACT_LAB_ID>
huggingface_token: <HF_TOKEN>
```

---

## 5. Robot profile (`<profiles-dir>/<name>.yaml`)

A robot profile captures hardware and policy metadata. Start from a skeleton — `oopsie-data new-profile --name <name>` writes one into `./robot_profiles/` — then ask the user the questions below and fill it in. Do not start from one of the bundled example profiles: they describe someone else's robot, and an unnoticed leftover field is recorded into every episode's HDF5 attrs and uploaded.

Make sure to list available options to the user where the choice is constrained to a set of options, and explain them if the user asks for additional detail.

### 5a. Robot & policy identity
| Question | YAML key | Example |
|---|---|---|
| What is the policy name? | `policy_name` | `pi0.5`, `act_plus_plus` |
| What is the robot name? | `robot_name` | `franka_droid`, `aloha` |
| What is the gripper name? | `gripper_name` | `robotiq_2f_85`, `aloha_gripper` |
| Is this a bimanual (dual-arm) setup? | `is_biarm` | `true` / `false` |
| Does the robot use a mobile base? | `uses_mobile_base` | `true` / `false` |
| What is the control frequency (Hz)? | `control_freq` | `10`, `50` |
| What are the camera names? (list) | `camera_names` | `[left, right, wrist]` |

### 5b. Observation space
| Question | YAML key | Options |
|---|---|---|
| Which robot state keys are recorded? | `robot_state_keys` | **`joint_position` and `gripper_position` are mandatory**; `cartesian_position` and `base_position` are optional additions |
| What are the joint names (in order)? | `robot_state_joint_names` | required — e.g. `joint_1 … joint_7` |
| If `cartesian_position` is included: what orientation representation does the robot state use? | `robot_state_orientation_representation` | `euler_xyz`, `quat`, `matrix`, `rot6d`, `rotvec` |

### 5c. Action space
| Question | YAML key | Options |
|---|---|---|
| What action types does the policy output? | `action_space` | Not a free choice — **exactly one arm action** from `joint_position`, `joint_velocity`, `cartesian_position`, `cartesian_velocity`; **at least one gripper action** from `gripper_position`, `gripper_velocity`, `gripper_binary`; **at most one base action** from `base_velocity`, `base_position`; no other keys. A profile declaring `uses_mobile_base: true` must include a base action. |
| What are the joint names for arm actions? | `action_joint_names` | same order as the action vector. **Required** whenever `joint_position` or `joint_velocity` is in the action space — not optional. |
| If `cartesian_position` is in the action space: what orientation representation? | `orientation_representation` | `euler_xyz`, `quat`, `matrix`, `rot6d`, `rotvec`. Applies to `cartesian_position` only — `cartesian_velocity` is recorded exactly as given, with no conversion and no shape check. |

### 5d. Optional keys
These are not required but can be stored for reproducibility:
- `controller` — e.g. `OSC`, `joint_position`, `joint_velocity`
- `gains` — controller gain parameters (see template)
- Camera intrinsic / extrinsic calibration matrices

---

## 6. Validate the config

Run the test suite to catch config errors early:

```bash
pytest oopsie_data_tools/test/
```

DO not modify the project as this can cause issues later on. Instead, ask the user to manually check issues and to contact the project team if necessary. It is vital that you do not change the code in the oopsie_data_tools directory, only templates and configs, without the user's expressed permission.

---

## 7. Choose a data collection workflow

Ask the user which workflow they need:

**A. In-the-loop** — annotate each episode right after it is collected (requires the annotation server to be running during robot operation).

**B. Bulk collection** — collect all episodes first, annotate later using the standalone annotation server.

For **A**, the annotation server will be launched as part of the robot script.

For **B**, run this command needs to be run after.
```bash
oopsie-data annotate \
  --samples-dir ./samples \
  --annotator-name <YOUR_NAME> \
  --port 5001
```

Omitting `--annotator-name` makes the command prompt for it. The equivalent long form is
`python -m oopsie_data_tools.annotation_tool.annotator_server --samples-dir ./samples
--annotator-name <YOUR_NAME> --port 5001`.

---

## 8. Integrate `EpisodeRecorder` into the robot script

Ask the user:
- Where is their robot control loop? (file path)
- What variable holds the robot observation dict? (must have `robot_state` and `image_observation` keys)
- What variable holds the action dict? (keys must match `action_space` in the robot profile)
- Where should episode HDF5 files and videos be saved? (`data_root_dir`)
- Who is running the evaluation? (`operator_name`, stamped into every episode)

Minimal integration pattern:
```python
from oopsie_data_tools.annotation_tool.episode_recorder import EpisodeRecorder
from oopsie_data_tools.utils.robot_profile.robot_profile import load_robot_profile

profile = load_robot_profile("<profiles-dir>/<your_profile>.yaml")
recorder = EpisodeRecorder(
    robot_profile=profile,
    data_root_dir="./samples",
    operator_name="<operator>",
    # resume_session_name="20260101_120000",  # optional: append to an existing session
)

for _ in range(num_episodes):
    recorder.reset_episode_recorder()  # clears the buffers between episodes

    # Inside the control loop:
    recorder.record_step(observation=obs, action=action)

    # After the rollout ends. `instruction` is required; `success` is optional
    # (omit it to leave the episode unannotated for the web annotator).
    recorder.finish_rollout(instruction="pick up the red block", success=success)
```

Verify that the keys are consistent between the robot profile and the ones passed for recording. The data validation will fail otherwise.

---

## 9. Upload data

After annotation is complete:
```bash
oopsie-data upload --path ./samples
```

This validates and pushes episodes to the lab-specific HuggingFace repository. To check the
data without uploading, run `oopsie-data validate --path ./samples` (or
`oopsie-data upload --path ./samples --skip-upload` for the full pre-upload check).

The `scripts/validate_and_upload/upload.py` and `validate.py` scripts remain available. They
forward their arguments straight to the CLI, so they accept the same flags — including the
older `--episode_id` / `--skip_validate` / `--skip_upload` spellings — and behave identically.

---

## Common mistakes to catch

- `lab_id` unset, blank (`lab_id:`), or still the placeholder in `contributor_config.yaml` → a clear `RuntimeError` (pointing to the registration form) at `EpisodeRecorder.__init__` and when running `oopsie-data upload`. Capitalisation must match exactly the value you were given.
- Config edited in the wrong place — e.g. editing the clone's `configs/` while `$OOPSIE_CONFIG_DIR` or `~/.config/oopsie-data` also exists, which take precedence. The error message names the file that was actually read.
- After uploading, run `python scripts/validate_and_upload/query_submissions.py` to confirm your episodes landed in the lab HuggingFace repo.
- Action dict keys not matching `action_space` in the robot profile → validation error at `record_step`.
- Joint-space actions require `joint_position` in `robot_state_keys`; Cartesian actions require `cartesian_position`.
- If `joint_position` is in `robot_state_keys`, `robot_state_joint_names` must be a non-empty list.
- `robot_state_joint_names` length not matching the `joint_position` array length → HDF5 schema error.
- Running `uv sync` without `--extra tfds` or `--extra droid` when those features are needed (note: those two extras conflict with each other).
- Passing an action chunk instead of per-step actions → caught only for `cartesian_position`, which is shape-checked to `(7,)` or `(14,)`. For joint action spaces a `(T, chunk, dof)` array passes both the DOF and trajectory-length checks and is recorded silently, so check this yourself.
- `cartesian_position` in state/action but `orientation_representation` not set → the value is recorded unconverted and then rejected unless it is already `[x, y, z, qx, qy, qz, qw]` with a unit quaternion. A representation that is set but does not match what the policy emits is reported by width, e.g. "QUAT orientation expects 4 value(s), got 3".
- `robot_state_joint_names` length not matching the `joint_position` array length → `EpisodeValidationError` raised inside `finish_rollout`, before any HDF5 file is written. (`EpisodeValidationError` subclasses `AssertionError`, so `except AssertionError` still catches it.)
- Expecting output from `scripts/dataset_conversion/` to validate directly. Those converters emit the legacy `robotic_failure_upload_data_format_v1` layout on purpose; run `scripts/migrate_hdf5_format.py` on the output first. The converters print the exact command.
- Running one of the `examples/inference_examples/` scripts without `--robot-profile`. It is required, and deliberately has no default.

## Important mistakes that will not raise an error

These need to be verified manually by the user.

- Action space not in absolute, but in delta coordinates
  - Delta coordinates cannot easily be processed by downstream applications as the base offset is not recorded
- Quaternion representation is in wrong order if passed explicitly
  - The framework provides a best effort test, but it cannot catch all edge-cases
