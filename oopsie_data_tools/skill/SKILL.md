---
name: oopsie-data
description: Use when working with Oopsie Data robotic manipulation failure datasets or the oopsie-data / oopsie-data-tools CLI — recording rollouts with EpisodeRecorder, writing or debugging a robot profile, annotating episodes and the failure taxonomy, reading or validating oopsiedata_format_v1 HDF5 (.h5) episode files, converting an existing robot dataset into the format, or uploading a lab's submission to HuggingFace.
---

# Oopsie Data

This skill is for using the toolkit: recording, annotating, validating and uploading a
contributor's data. 

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

Some users prefer to convert pre-recorded data instead of recording through `EpisodeRecorder`.
That path writes the HDF5 directly rather than going through the recorder, so it bypasses every
recording-time check — see `reference/conversion.md`.

Supporting commands: `show-config` (which config files are in effect), `submissions` (what the
lab has already uploaded), `inspect <file.h5>` (dump one episode's structure — a debugging aid
that works even on files `validate` rejects; the path is positional, there is no `--path`),
`restructure` (split a directory HuggingFace would reject), `install-skill` (copy this skill
into an agent's skills directory).

Inside a git checkout that was set up with `uv sync`, prefix every command with `uv run`. A
pip- or uv-installed package puts `oopsie-data` on `PATH` directly.

## Machine-readable output

`validate`, `show-config` and `inspect` each take `--json`, which prints a structure on stdout
instead of prose and logs nothing else. Prefer it: it is the difference between reading a
result and parsing one. Exit codes are identical either way.

```bash
oopsie-data validate --path ./samples --json     # one record per episode: passed, error
oopsie-data show-config --json                   # both search chains and what won
oopsie-data inspect ./samples/000000.h5 --json   # the full group/dataset/attr tree
```

`validate --json` gives `episodes[]` with `episode`, `path`, `passed`, and on failure `error`
and an `error_type` of `validation` (the episode is bad — the user's to fix) or `unexpected`
(the validator broke — report it, do not work around it), plus `total`/`passed`/`failed`
counts. That is what to use when triaging a directory rather than checking a single file.
`show-config --json` masks the token unless `--show-token` is given, so it is safe to quote
back to the user.

## Commands that block or prompt

Three commands are the human's to run, and running them in the foreground yourself stalls the
session or fails for a reason that has nothing to do with the data.

**`annotate` never returns.** It serves the UI until Ctrl-C, by design. Start it in the
background and give the user the URL, or hand them the command to run — do not wait on it.
Annotating is a human judgement about what happened in a rollout, so there is nothing for you
to do while it runs anyway.

**`restructure` prompts for confirmation** and has no terminal check, so without `--yes` it
either hangs waiting for input or dies on `EOFError`. It copies the whole dataset, which is
why the prompt exists — get the user's agreement, then pass `--yes`. `upload
--with-restructure` treats the flag itself as that agreement and does not prompt.

**`init` is interactive**, but it checks for a terminal and exits 1 with an explanation
rather than hanging. It runs unattended only when `--lab-id` and `--hf-token` are both given —
and a token passed as a flag lands in the user's shell history, so prefer letting them run
`init` themselves. The same applies to `annotate --annotator-name`: omitted, it prompts;
without a terminal it errors out.
You can run it, but ask the user to provide the values and pass them as flags rather than typing them yourself.

**`upload`**: out of principle, you should not run it for the user. It publishes their data to the submission repository, so
it is best practice to let the user run it themselves. If you do run it, confirm with them first, and pass `--skip-upload` to do a dry run that validates everything without publishing. Only run it
in full if the user explicitly asks and insists.

Everything else — `validate`, `show-config`, `inspect`, `submissions`, `new-profile`,
`install-skill` — runs to completion on its own and is safe to invoke directly.

## Rules

**Never invent identity or annotation content.** `lab_id` comes from the registration form,
`operator_name` and `annotator_name` are the human's, and an episode's `outcome` and
`episode_description` record what actually happened in a rollout. If one is missing, ask. Do
not fill in a plausible value.

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

Read the one you need; do not read all five.

- `reference/setup.md` — installing, where configs live, `init`, choosing an annotation
  workflow, and wiring `EpisodeRecorder` into a robot control loop. The onboarding path.
- `reference/robot-profile.md` — every profile field, which are required, the legal values,
  and the questions to ask the user when filling one in.
- `reference/format.md` — the `oopsiedata_format_v1` HDF5 layout and every rule the validator
  enforces. Read before theorizing about why an episode was rejected.
- `reference/conversion.md` — the same schema from the writing side, for converting an existing
  dataset into `oopsiedata_format_v1` instead of recording it.
- `reference/troubleshooting.md` — common errors, and the mistakes that pass validation
  silently and must be checked by a human.

## Reference

- Documentation: <https://oopsie-data.com>
- Toolkit source: <https://github.com/oopsie-data/oopsie-data-tools>
- `oopsie-data --help` lists every command; `oopsie-data <command> --help` documents its flags
  and shows worked examples.
