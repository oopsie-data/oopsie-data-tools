---
name: oopsie-data
description: Use when working with Oopsie Data robotic manipulation failure datasets — recording rollouts, annotating episodes, validating the HDF5 format, or uploading a submission with the oopsie-data CLI.
---

# Oopsie Data

Oopsie Data is a community dataset of **robotic manipulation failures**. Contributors record
rollouts, annotate what went wrong, and upload them to a shared HuggingFace repo. The
`oopsie-data` CLI is the single entry point for all of it — there are no side scripts.

Run `oopsie-data <command> --help` before constructing an invocation rather than guessing at
flags.

## The workflow

Once per contributor, then once per recording session:

```bash
oopsie-data init                              # lab id + HuggingFace token (one time)
oopsie-data new-profile                       # robot profile skeleton, then fill it in by hand
#           ... record episodes by calling EpisodeRecorder from the robot control loop
oopsie-data annotate --samples-dir ./samples  # label episodes in a browser UI
oopsie-data validate --path ./samples         # check against oopsiedata_format_v1
oopsie-data upload   --path ./samples         # validates again, then publishes
```

Recording is the step that produces the episodes everything after it consumes; it is code the
user adds to their own control loop, not a CLI command. See `reference/setup.md`.

Note that some users prefer to convert pre-recorded data. Look for general instructions in `references/setup.md` and `references/conversion.md`.

Supporting commands: `show-config` (which config files are in effect), `submissions` (what the
lab has already uploaded), `inspect <file.h5>` (dump one episode's structure — a debugging aid
that works even on files `validate` rejects; the path is positional, there is no `--path`),
`restructure` (split a directory HuggingFace would reject), `install-skill` (copy this skill
into a Claude configuration).

Inside a git checkout that was set up with `uv sync`, prefix every command with `uv run`. A
pip- or uv-installed package puts `oopsie-data` on `PATH` directly.

## Rules

**Never invent identity or annotation content.** `lab_id` comes from the registration form,
`operator_name` and `annotator_name` are the human's, and failure descriptions describe what
actually happened in a rollout. If one is missing, ask. Do not fill in a plausible value.

**Never hand-write `contributor_config.yaml`.** Run `oopsie-data init`. It rejects the
`your_lab_id` placeholder, writes the file mode 0600, and keeps the token out of the checkout
by defaulting to the per-user config directory. Writing the YAML yourself loses all three.

**Uploads are public and effectively irreversible.** `oopsie-data upload` publishes to
`OopsieData-Submissions/<lab_id>` on HuggingFace. Confirm with the user before running it, even
if they asked for the whole pipeline in one go. `--skip-upload` runs every check and publishes
nothing; use it as the dry run.

**Never reach for `--skip-validate`.** Validation is the only thing standing between a
malformed episode and the shared dataset. If validation fails, ask the user how to fix the
data. Do not assume you know better than the validator or the user.

**A robot profile that has not been filled in will not load, by design.** `new-profile` writes
a skeleton whose required fields are blank so a half-edited profile cannot stamp placeholder
metadata into recorded episodes. If loading one raises, fill in the fields; do not work around
the loader.

**Do not change code under `oopsie_data_tools/` to make a check pass.** Configs, robot profiles
and the user's own robot script are yours to edit; the toolkit is not, without the user's
explicit permission. A failing validator is a finding to report, not an obstacle to remove.

**Credentials and robot profiles are different things** with separate lookup chains.
`contributor_config.yaml` belongs to the person, robot profiles belong next to the robot code.
`oopsie-data show-config` prints both chains and marks what is actually in use — start any
config confusion there, and see `reference/setup.md` for the resolution order.

## Reference files

Read the one you need; do not read all four.

- `reference/setup.md` — installing, where configs live, `init`, choosing an annotation
  workflow, and wiring `EpisodeRecorder` into a robot control loop. The onboarding path.
- `reference/robot-profile.md` — every profile field, which are required, the legal values,
  and the questions to ask the user when filling one in.
- `reference/format.md` — the `oopsiedata_format_v1` HDF5 layout and every rule the validator
  enforces. Read before theorizing about why an episode was rejected.
- `reference/troubleshooting.md` — common errors, and the mistakes that pass validation
  silently and must be checked by a human.

## Reference

- Documentation: <https://oopsie-data.com>
- Toolkit source: <https://github.com/oopsie-data/oopsie-data-tools>
- `oopsie-data --help` lists every command; `oopsie-data <command> --help` documents its flags
  and shows worked examples.
