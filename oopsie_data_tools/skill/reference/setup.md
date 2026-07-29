# Setup

Onboarding a user into `oopsie-data-tools`, in order. Ask the questions in each section before
writing anything.

## 1. Prerequisites

- Python 3.8 or newer. 3.8 is the supported floor and is tested in CI alongside 3.10 and 3.12,
  but 3.8 itself has been end-of-life since October 2024 — prefer newer if the user has a choice.
- `uv` (preferred) or `pip`.
- A **lab ID** and **HuggingFace token** from the registration form,
  <https://forms.gle/9arwZHAvRjvbozoT7>. Nothing works without these; if the user does not have
  them, send them there first.

## 2. Installation

To use the toolkit:

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

`droid` is the only extra; everything the CLI does works off a bare install. In a `uv sync`
checkout, every command below needs a `uv run` prefix.

Confirm `oopsie-data --version` works before continuing.

## 3. Where configs live

The two kinds of config are looked up through **separate chains**, because they belong to
different things. In each chain, the first location that exists wins.

**Credentials** (`contributor_config.yaml`) usually belong to the user and are shared by every
project:

1. `$OOPSIE_CONFIG_DIR` — explicit override
2. `.` or `./configs`, relative to the working directory
3. `~/.config/oopsie-data` (or `$XDG_CONFIG_HOME/oopsie-data`)

The project-local legs allow a per-project identity — a second lab, a shared machine — and
win over the per-user one. `oopsie-data init` offers both and defaults to the per-user
directory, because a config in a project holds a token where a commit can pick it up.

**Robot profiles** belong to the robot code that loads them, and are *never* looked up in the
user config directory:

1. `$OOPSIE_ROBOT_PROFILES_DIR` — explicit override
2. `./robot_profiles` or `./configs/robot_profiles`, relative to the working directory

Neither chain treats a source checkout as special. Working inside a clone, its
`configs/robot_profiles` is found by the ordinary cwd-relative leg and nothing else about it
is privileged.

So: let `oopsie-data init` write the contributor config (it defaults to `~/.config/oopsie-data`
over the working directory, because the file holds a token and a token under version control
can be committed). Put the robot profile next to the user's robot code — a `robot_profiles/` directory
beside their eval script is the normal choice — and load it by explicit path.

To keep either somewhere custom, persist it in the shell rc file:

```bash
echo 'export OOPSIE_CONFIG_DIR=/path/to/oopsie-config' >> ~/.bashrc          # or ~/.zshrc
echo 'export OOPSIE_ROBOT_PROFILES_DIR=/path/to/profiles' >> ~/.bashrc
```

`--config-dir <dir>` overrides the credential location for one run. It is a flag on
`oopsie-data` itself, so it goes **before** the subcommand — `oopsie-data --config-dir <dir>
upload --path ...`. Placing it after the subcommand is an argparse error.

When unsure what is in effect, run `oopsie-data show-config`: it prints both chains, the
location that wins, and the lab id and token in use (`--show-token` prints the token unmasked).
`$HF_TOKEN` in the environment overrides the stored token.

Below, `<config-dir>` means the resolved credential location and `<profiles-dir>` the resolved
profile location.

## 4. Contributor config — `oopsie-data init`

Ask the user for:

1. **Their lab ID** — the exact string from registration. Capitalization matters; a wrong value
   blocks access to the lab's HuggingFace repo.
2. **Their HuggingFace token.**

Then have them run:

```bash
oopsie-data init
```

It asks which config directory to use, then for the lab id and token, rejects the `your_lab_id`
placeholder, and writes `<config-dir>/contributor_config.yaml` with mode 0600. `--lab-id`,
`--hf-token`, `--no-verify-token` and `--force` skip the corresponding prompts, so it also runs
unattended.

Do **not** write this file by hand. The token check is **advisory**: a token that fails to
verify is reported as a warning, saved anyway, and `init` still exits 0. Only the lab id is
actually rejected. You find out a token is wrong at `oopsie-data upload`.

`init` does not create robot profiles — see section 5.

## 5. Robot profile

```bash
oopsie-data new-profile --name <name>       # writes <profiles-dir>/<name>.yaml
```

Then fill it in by hand, using `reference/robot-profile.md` for the questions to ask and the
legal values. Do not start from one of the bundled example profiles: they describe someone
else's robot, and an unnoticed leftover field is recorded into every episode's HDF5 attrs and
uploaded.

Check the result loads:

```bash
python -c "from oopsie_data_tools.utils.robot_profile.robot_profile import load_robot_profile; load_robot_profile('<path>')"
```

## 6. Confirm the setup

`oopsie-data show-config` is what verifies a contributor config; loading the profile as above is
what verifies a profile. Real confirmation comes from `oopsie-data validate --path ./samples`
once episodes exist.

Running `pytest oopsie_data_tools/test/` checks the toolkit, not the user's setup — the suite
deliberately isolates `HOME`, `XDG_CONFIG_HOME`, both `OOPSIE_*` variables and the working
directory, so it cannot see the user's config at all.

## 7. Choose an annotation workflow

Ask which the user needs:

**A. In-the-loop** — annotate each episode right after it is collected. The annotation server
runs during robot operation; launch it from the robot script (`WebRolloutAnnotator`), or run
`oopsie-data annotate --with-rollouts`.

**B. Bulk collection** — collect everything first, annotate later:

```bash
oopsie-data annotate --samples-dir ./samples --annotator-name <YOUR_NAME> --port 5001
```

`--samples-dir` defaults to `./samples` and `--port` to 5001. Omitting `--annotator-name`
prompts for it, and errors out when stdin is not a terminal. `--no-browser` suppresses opening
a browser. The equivalent long form is `python -m
oopsie_data_tools.annotation_tool.annotator_server --samples-dir ./samples --annotator-name
<YOUR_NAME> --port 5001`.

## 8. Wire `EpisodeRecorder` into the robot script

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

    # After the rollout ends. `instruction` is required; `success` is optional
    # (omit it to leave the episode for the web annotator — but note that an episode
    # with no annotation at all fails `oopsie-data validate`, so it must be annotated
    # before it can be uploaded).
    recorder.finish_rollout(instruction="pick up the red block", success=success)
```

`record_step` does not mutate the dicts it is passed. `finish_rollout` validates *before*
writing anything, so a rejected episode leaves no MP4s and no HDF5 on disk.

The keys in the profile and the keys passed to `record_step` must agree exactly — see
`reference/format.md`.

## 9. Upload

```bash
oopsie-data upload --path ./samples
```

This validates, then pushes to `OopsieData-Submissions/<lab_id>`. To check without publishing,
use `oopsie-data validate --path ./samples`, or `upload --path ./samples --skip-upload` for the
full pre-upload sequence. `--episode-id` restricts either command to one episode; `--log-path`
writes a report.

If upload refuses because a directory holds more than 10,000 files (a HuggingFace limit), add
`--with-restructure`:

```bash
oopsie-data upload --path ./samples --with-restructure
```

That writes a restructured copy to `./samples_restructured` and uploads it. Every directory over
the limit — at any depth — is split into numbered subfolders of 500 episodes; the rest of the
tree is copied through unchanged, with video paths inside the copied HDF5 files rewritten. It
copies rather than moves, so you need room for a second copy, and the original is untouched
until the user deletes it. Run `oopsie-data restructure --source ./samples --output <dir>`
separately to put the copy elsewhere.

Afterwards, `oopsie-data submissions` confirms what landed in the lab's repo.
