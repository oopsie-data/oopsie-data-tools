---
name: oopsie-data
description: Use when working with Oopsie Data robotic manipulation failure datasets or the oopsie-data / oopsie-data-tools CLI — recording rollouts with EpisodeRecorder, writing or debugging a robot profile, annotating episodes and the failure taxonomy, reading or validating oopsiedata_format_v1 HDF5 (.h5) episode files, converting an existing robot dataset into the format, or uploading a lab's submission to HuggingFace.
---

# Oopsie Data

Oopsie Data is a community dataset of **robotic manipulation failures**. Contributors record
rollouts, annotate what went wrong, and upload them to a shared HuggingFace repo. The
`oopsie-data` CLI is the single entry point — there are no side scripts. Run
`oopsie-data <command> --help` rather than guessing at flags.

## The workflow

`init` and `new-profile` are one-time setup. `annotate`, `validate`, and `upload` are the main loop.

```bash
oopsie-data init                              # lab id + HuggingFace token (one time)
oopsie-data new-profile                       # robot profile skeleton, then fill it in by hand
#           ... record episodes by calling EpisodeRecorder from the robot control loop
oopsie-data annotate --samples-dir ./samples  # label episodes in a browser UI
oopsie-data validate --path ./samples         # check against oopsiedata_format_v1
oopsie-data upload   --path ./samples         # validates again, then publishes
```

Recording is code the user adds to their own control loop, not a CLI command — see
`reference/setup.md`. Converting pre-recorded data instead writes the HDF5 directly and so
bypasses every recording-time check — see `reference/conversion.md`.

Also: `show-config`, `submissions` (counts, total size and when the last upload landed, all
from Hub metadata — nothing is downloaded), `inspect <file.h5>` (structure dump; works even on files
`validate` rejects, and the path is positional), `restructure` (split a directory HuggingFace
would reject), `install-skill`. `validate`, `show-config` and `inspect` take `--json`; prefer it.
Two things the payloads do not tell you: a failed episode's `error_type` is either `validation`
(the episode is bad — the user's to fix) or `unexpected` (the validator broke — report it, do not
work around it), and `inspect`'s `robot_profile` attr is a JSON string *inside* the JSON, so it
needs a second parse. `inspect` output is large — narrow it before reading it.

Inside a `uv sync` checkout, prefix every command with `uv run`.

## Commands that block or prompt

- **`annotate` never returns** — it serves the UI until Ctrl-C. Background it and give the user
  the URL, or hand them the command. Annotation is human judgement; there is nothing for you to
  do while it runs.
- **`restructure` prompts and has no terminal check** — without `--yes` it hangs or dies on
  `EOFError`. It copies the whole dataset, which is why it asks. `upload --with-restructure`
  treats the flag as that agreement.
- **`init` prompts** (it exits 1 rather than hanging). It runs unattended with `--lab-id` and
  `--hf-token`, but a token in a flag lands in shell history — prefer letting the user run it.
- **Do not run `upload` for the user.** It publishes their data to the shared repo. Hand them the
  command. If you must run it, confirm first and pass `--skip-upload` to validate without
  publishing — only run it in full if the user explicitly insists.

## Rules

**Never invent identity or annotation content.** `lab_id` comes from the registration form,
`operator_name` and `annotator_name` are the human's, and `outcome` and `episode_description`
record what actually happened in a rollout. If one is missing, ask.

**Never hand-write `contributor_config.yaml`.** `oopsie-data init` rejects the `your_lab_id`
placeholder, writes mode 0600, and keeps the token out of the checkout. Writing it yourself
loses all three.

**If validation fails, fix the data or report it — never `--skip-validate`.** Same for a robot
profile skeleton that will not load: its required fields are blank on purpose, so that a
half-edited profile cannot stamp placeholder metadata into every episode.

## Reference files

Read the one you need; do not read all five.

- `reference/setup.md` — installing, where configs live, `init`, and wiring `EpisodeRecorder`
  into a control loop. The onboarding path.
- `reference/robot-profile.md` — profile fields, legal values, and the questions to ask.
- `reference/format.md` — the HDF5 layout and what the validator enforces. Read before
  theorizing about a rejection.
- `reference/conversion.md` — the same schema from the writing side, for converting an existing
  dataset instead of recording it.
- `reference/troubleshooting.md` — common errors, and the mistakes that pass validation silently.

Docs: <https://oopsie-data.com> · Source:
<https://github.com/oopsie-data/oopsie-data-tools>
