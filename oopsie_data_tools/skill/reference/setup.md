# Setup

Onboarding a user into `oopsie-data-tools`, in order. Ask the questions in each section before
writing anything.

## 1. Install

Needs Python 3.8+ (3.10+ preferred), and a **lab ID** and **HuggingFace token** from the
registration form, <https://forms.gle/9arwZHAvRjvbozoT7>. Nothing works without those.

```bash
uv add oopsie-data-tools        # or: pip install oopsie-data-tools
```

To work from a checkout (contributing to the toolkit, or running the bundled examples):

```bash
git clone https://github.com/oopsie-data/oopsie-data-tools
cd oopsie-data-tools
uv sync                         # or: pip install -e .
uv sync --extra droid           # only for examples/inference_examples/
```

In a `uv sync` checkout, every command below needs a `uv run` prefix. Confirm
`oopsie-data --version` works before continuing.

Warning: the `opencv-python` floor is `4.6.0.66`, low enough to fit into an existing robot
environment. Versions below `4.10.0.84` are built against numpy 1 and raise `ImportError:
numpy.core.multiarray failed to import` under numpy 2, and opencv declares no upper bound that
would let a resolver avoid that pairing. If the environment has numpy 2, it needs
`opencv-python>=4.10.0.84`.

## 2. Where configs live

The two kinds are looked up through **separate chains**. In each, the first location that exists
wins.

**Credentials** (`contributor_config.yaml`) belong to the user and are shared across projects:

1. `$OOPSIE_CONFIG_DIR`
2. `.` or `./configs`, relative to the working directory
3. `~/.config/oopsie-data` (or `$XDG_CONFIG_HOME/oopsie-data`)

**Robot profiles** belong to the robot code, they are looked up at:

1. `$OOPSIE_ROBOT_PROFILES_DIR`
2. `./robot_profiles` or `./configs/robot_profiles`, relative to the working directory

`oopsie-data show-config` prints both chains and what won. 

Warning: `$HF_TOKEN` in the environment
overrides the stored token. `--config-dir <dir>` overrides the credential location for one run,
and is a flag on `oopsie-data` itself — it goes **before** the subcommand.

Below, `<config-dir>` and `<profiles-dir>` mean the resolved locations.

## 3. Contributor config — `oopsie-data init`

Ask the user for their **lab ID** (the exact string from registration — capitalization matters)
and their **HuggingFace token**, then have them run `oopsie-data init`.

It prompts for a config directory, lab id and token, and
writes `<config-dir>/contributor_config.yaml` with mode 0600. `--lab-id`, `--hf-token`,
`--no-verify-token` and `--force` skip the corresponding prompts.

The token check is **advisory**: a token that fails to verify is warned about, saved anyway, and
`init` still exits 0. You find out a token is wrong at `oopsie-data upload`.

## 4. Robot profile

```bash
oopsie-data new-profile --name <name>       # writes <profiles-dir>/<name>.yaml
```

Fill it by prompting the user for input, using `reference/robot-profile.md` for the questions and legal values. 

Check it loads:

```bash
python -c "from oopsie_data_tools.utils.robot_profile.robot_profile import load_robot_profile; load_robot_profile('<path>')"
```

## 5. Choose an annotation workflow

Ask which the user needs:

**A. In-the-loop** — annotate each episode right after it is collected. Launch the annotation
server from the robot script (`WebRolloutAnnotator`), or run `oopsie-data annotate
--with-rollouts`.

**B. Bulk collection** — collect everything first, annotate later:

```bash
oopsie-data annotate --samples-dir ./samples --annotator-name <YOUR_NAME> --port 5001
```

`--samples-dir` defaults to `./samples` and `--port` to 5001. Omitting `--annotator-name`
prompts for it. `--no-browser` suppresses opening a browser.

If the user specifies that they have data already, consult `reference/conversion.md` for information on how to convert existing data.

## 6. Wire `EpisodeRecorder` into the robot script

Ask the user:

- Where is their robot control loop? (file path)
- What variable holds the observation dict? (needs `robot_state` and `image_observation` keys)
- What variable holds the action dict? (keys must equal `action_space` in the profile exactly)
- Where should episode HDF5 files and videos be saved? (`data_root_dir`)
- Who is running the evaluation? (`operator_name`, stamped into every episode)

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

    # After the rollout. `instruction` is required; `success` is optional — omit it to leave
    # the episode for the web annotator, but an episode with no annotation at all fails
    # `oopsie-data validate`, so it must be annotated before it can be uploaded.
    recorder.finish_rollout(instruction="pick up the red block", success=success)
```

`finish_rollout` validates *before* writing anything, so a rejected episode leaves no MP4s and
no HDF5 on disk. The keys in the profile and the keys passed to `record_step` must agree exactly
— see `reference/format.md`.

