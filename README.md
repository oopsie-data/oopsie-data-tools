# Oopsie Tools

Tools for collecting, annotating, inspecting, and converting robotic manipulation rollout data.

This repository currently provides tools for:

- HDF5 episode recording (`EpisodeRecorder`)
- Web annotation workflows
- In-the-loop annotation during policy rollout

as well as all the necessary utilities to validate, inspect, and upload Oopsie-Data to the official repositories.

[Sign up today](https://forms.gle/9arwZHAvRjvbozoT7) and start contributing!

---

For detailed explanations on how to use our tooling and contribute to the project, please visit [our website](https://oopsie-data.com/).

For an overview of the steps necessary to integrate the tooling into your workflow and to contribute data to the official Oopsie Data repositories, check out [our quickstart guide](https://oopsie-data.com/quickstart).

## Requirements

The tooling is tested with Python versions 3.8 - 3.12.

> **Warning:** For compatibility with droid utilties, we require `opencv-python>-4.6.0.66`. This version is built against numpy 1 and raises
> `ImportError: numpy.core.multiarray failed to import` under numpy 2. opencv declares no upper
> numpy bound before 5.0, so pip and uv will happily install that broken pairing without
> reporting a conflict. If your environment uses numpy 2, require `opencv-python>=4.10.0.84`.

## Repository structure

The main tooling for data gathering and annotation is located in `oopsie-data-tools`.

We provide example scripts for automatically collecting and annotating evaluation data while running policy inference in examples. Currently we support the evaluation scripts supported by `openpi` and Trossen robotics `act_plus_plus` repository. If you want to run a different evaluation script, check out the detailed instructions on integrating our tools into standard robot evaluation pipelines.

## Issue/PR for support requests

We are very aware that changing eval code and recording data while doing experiments can be a big hassle and cause friction. We are therefore happy to help you integrate the recording tool into your setup. Please let us know what robot platform and policy you are evaluating and where to find a sample of your evaluation & inference code.

## Contributing

You can use our toolset any time you like to record and annotate robot rollouts. To contribute your data to the official Oopsie Dataset, please follow the [sign-up instructions](https://oopsie-data.com/contributing/)!

If you run into any issues, please do not hesitate to contact the team via mail or raise an issue here.
