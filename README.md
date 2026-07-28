# Oopsie Tools

Tools for collecting, annotating, inspecting, and converting robotic manipulation rollout data.

This repository currently provides around:

- HDF5 episode recording (`EpisodeRecorder`)
- Web annotation workflows
- In-the-loop annotation during policy rollout

## Command line tool

Installing the package (`uv sync`, or `pip install -e .`) provides the `oopsie-data` command:

```bash
oopsie-data init                                 # first-time setup (lab id + HF token)
oopsie-data show-config                          # where configs are read from, and what is in them
oopsie-data new-profile                          # starter robot profile to fill in
oopsie-data annotate --samples-dir ./samples     # launch the web annotation UI
oopsie-data validate --path ./samples            # check episodes against the schema
oopsie-data upload   --path ./samples            # validate, then upload to HuggingFace
```

`init` prompts for your lab id and HuggingFace token, verifies the token, and saves them to a
location the other commands find automatically. `annotate` asks for your annotator name if
`--annotator-name` is not passed.

Credentials and robot profiles are looked up separately, so the tool also works when
installed outside a checkout of this repository.

`contributor_config.yaml` belongs to you and is shared by every project:

1. `$OOPSIE_CONFIG_DIR`
2. `~/.config/oopsie-data` (or `$XDG_CONFIG_HOME/oopsie-data`)
3. this repository's `configs/` directory

Robot profiles belong to the robot code that uses them, and are never expected in your user
config directory:

1. `$OOPSIE_ROBOT_PROFILES_DIR`
2. `./robot_profiles` or `./configs/robot_profiles`, relative to where you run the command
3. this repository's `configs/robot_profiles`

In practice a profile is usually loaded by explicit path — `load_robot_profile("path/to.yaml")`.
Nothing is ever read out of the installed package: config that lives in `site-packages` is
read-only, invisible to you, and would silently shadow your own. Run `oopsie-data new-profile` to
write a starter profile into `./robot_profiles/` and fill it in from there. Pass
`oopsie-data --config-dir <dir> <command>` to override the credential location for a single run.

Run `oopsie-data show-config` to see which of these locations is actually being used, along with
the lab id and token in effect.

### Upgrading: `configs/contributor_config.yaml` is no longer tracked

That file holds your HuggingFace token, and it used to be committed to the repository — which
meant `.gitignore` did not apply to it and your token showed up in `git status`. It is now
untracked and ignored.

If you have an existing clone with your credentials filled in, `git pull` may refuse:

```
error: Your local changes to the following files would be overwritten by merge:
        configs/contributor_config.yaml
```

Your credentials are safe; git is just protecting the file. Move them out of the working tree:

```bash
oopsie-data init            # writes to ~/.config/oopsie-data, verifies nothing is lost
rm configs/contributor_config.yaml
git pull
```

Nothing else changes — that location is still searched, so an existing setup keeps working. You
will see a warning pointing here until you move it.

---

For detailed explanations on how to use our tooling and contribute to the project, please visit [our website](https://oopsie-data.com/).

For an overview of the steps necessary to integrate the tooling into your workflow and to contribute data to the official Oopsie Data repositories, check out [our quickstart guide](https://oopsie-data.com/quickstart).
You can also use the information in AI_CONTEXT.md to guide a coding agent through the setup.

## Repository structure

The main tooling for data gathering and annotation is located in `oopsie-data-tools`.

We provide example scripts for automatically collecting and annotating evaluation data while running policy inference in examples. Currently we support the evaluation scripts supported by `openpi` and Trossen robotics `act_plus_plus` repository. If you want to run a different evaluation script, check out the detailed instructions on integrating our tools into standard robot evaluation pipelines.

## Issue/PR for support requests

We are very aware that changing eval code and recording data while doing experiments can be a big hassle and cause friction. We are therefore happy to help you integrate the recording tool into your setup. Please let us know what robot platform and policy you are evaluating and where to find a sample of your evaluation & inference code.

## Contributing

You can use our toolset any time you like to record and annotate robot rollouts. To contribute your data to the official Oopsie Dataset, please follow the [sign-up instructions](https://oopsie-data.com/contributing/)!

If you run into any issues, please do not hesitate to contact the team via mail or raise an issue here.
