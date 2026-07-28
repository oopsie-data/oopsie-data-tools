"""Tests for the three subcommands folded in from the old ``scripts/`` directory.

``inspect``, ``restructure`` and ``submissions`` used to be standalone scripts that nothing
tested and that a pip-installed user could not reach at all. They are subcommands now, so
they get the same coverage as the rest of the CLI. Nothing here touches the network.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import pytest

from oopsie_data_tools.cli import main
from oopsie_data_tools.test.fixtures.make_valid import write_valid_episode
from oopsie_data_tools.utils import hf_upload, restructure


@pytest.fixture
def info_log(caplog):
    """caplog at INFO.

    ``cli.main`` calls ``logging.basicConfig``, which is a no-op once pytest's logging
    plugin has attached its handler, so the root logger stays at WARNING and every
    ``logger.info`` line these commands emit as their *result* would be dropped.
    """
    caplog.set_level(logging.INFO)
    return caplog


# ── inspect ───────────────────────────────────────────────────────────────────


def test_inspect_dumps_groups_datasets_and_attrs(tmp_path, capsys):
    episode = write_valid_episode(tmp_path, "ep_a")

    assert main(["inspect", str(episode)]) == 0

    out = capsys.readouterr().out
    assert "[group] /" in out
    assert "observations/robot_states" in out
    assert "[dataset]" in out
    assert "'schema'" in out, "root attrs must be shown"


def test_inspect_works_on_a_file_validate_would_reject(tmp_path, capsys):
    """It is a structure dump, not a validator — that is the whole point of having it."""
    broken = tmp_path / "broken.h5"
    with h5py.File(broken, "w") as f:
        f.attrs["schema"] = "something_else_entirely"
        f.create_dataset("stray", data=[1, 2, 3])

    assert main(["inspect", str(broken)]) == 0
    assert "stray" in capsys.readouterr().out


def test_inspect_reports_a_missing_file_rather_than_raising(tmp_path, caplog):
    assert main(["inspect", str(tmp_path / "nope.h5")]) == 1
    assert "Not a file" in caplog.text


# ── restructure ───────────────────────────────────────────────────────────────


def test_restructure_does_nothing_when_the_folder_is_small_enough(tmp_path, info_log):
    write_valid_episode(tmp_path, "ep_a")

    assert main(["restructure", "--source", str(tmp_path), "--yes"]) == 0

    assert "No restructuring needed" in info_log.text
    assert not (tmp_path.parent / f"{tmp_path.name}_restructured").exists()


def test_restructure_copies_episodes_and_rewrites_video_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(restructure, "FILE_LIMIT", 1)
    source = tmp_path / "session"
    source.mkdir()
    write_valid_episode(source, "ep_a")
    write_valid_episode(source, "ep_b")
    output = tmp_path / "out"

    assert main(
        ["restructure", "--source", str(source), "--output", str(output), "--yes"]
    ) == 0

    # Source is untouched — this command must never be the thing that loses data.
    assert (source / "ep_a.h5").exists()
    assert (source / "ep_a_front.mp4").exists()

    copies = sorted(output.rglob("*.h5"))
    assert [p.name for p in copies] == ["ep_a.h5", "ep_b.h5"]
    for h5_path in copies:
        with h5py.File(h5_path, "r") as f:
            stored = f["observations/video_paths/front"][()]
        rel = stored.decode() if isinstance(stored, bytes) else stored
        assert "/" not in rel, "videos are copied flat next to their episode"
        assert (h5_path.parent / rel).is_file(), "the rewritten path must resolve"


def test_unique_video_dest_separates_two_videos_that_want_the_same_name(tmp_path):
    """Two distinct videos claiming one name must not collapse onto it.

    This used to return the colliding name regardless, so the copy silently overwrote the
    first episode's video. Exercised directly: ``restructure()`` collects HDF5 files from
    the source root only, where stems are unique by construction, so this branch is
    unreachable through the command and would otherwise never be tested.
    """
    used: dict[str, Path] = {}
    first = tmp_path / "one" / "ep_front.mp4"
    second = tmp_path / "two" / "ep_front.mp4"

    a = restructure._unique_video_dest(first, "front", "ep", used)
    b = restructure._unique_video_dest(second, "front", "ep", used)

    assert a == "ep_front.mp4"
    assert b != a, "a second, different video must not take the first one's name"
    # Idempotent: re-asking for a video already registered returns the same name, so a
    # caller may ask more than once per file without inventing new copies.
    assert restructure._unique_video_dest(first, "front", "ep", used) == a


def test_restructure_copies_a_shared_video_once(tmp_path, monkeypatch):
    """Two episodes referencing one video share the copy rather than duplicating it."""
    monkeypatch.setattr(restructure, "FILE_LIMIT", 1)
    source = tmp_path / "session"
    source.mkdir()
    write_valid_episode(source, "ep_a")
    write_valid_episode(source, "ep_b")
    # Point ep_b at ep_a's video, the way a re-annotated episode would.
    with h5py.File(source / "ep_b.h5", "r+") as f:
        del f["observations/video_paths/front"]
        f["observations/video_paths"].create_dataset(
            "front", data="ep_a_front.mp4", dtype=h5py.string_dtype(encoding="utf-8")
        )
    output = tmp_path / "out"

    assert main(
        ["restructure", "--source", str(source), "--output", str(output), "--yes"]
    ) == 0

    for h5_path in sorted(output.rglob("*.h5")):
        with h5py.File(h5_path, "r") as f:
            stored = f["observations/video_paths/front"][()]
        rel = stored.decode() if isinstance(stored, bytes) else stored
        assert (h5_path.parent / rel).is_file(), "every rewritten path must resolve"


def test_restructure_splits_a_nested_oversized_directory(tmp_path, monkeypatch, info_log):
    """Detection is recursive, so repair must be too.

    This used to detect the nested directory, restructure the *root* instead, and report
    success — leaving the upload just as blocked as before.
    """
    # Limit 2 puts the root exactly at the limit (one .h5 + one .mp4) while the nested
    # directory is over it. This is the shape that used to misfire: the old code saw an
    # oversized directory, then acted on the root because that is where it looked for
    # episodes.
    monkeypatch.setattr(restructure, "FILE_LIMIT", 2)
    source = tmp_path / "session"
    nested = source / "day_two"
    nested.mkdir(parents=True)
    write_valid_episode(source, "ep_root")
    write_valid_episode(nested, "ep_nested_a")
    write_valid_episode(nested, "ep_nested_b")
    output = tmp_path / "out"

    assert restructure.oversized_dirs(source) == [(nested, 4)], "only the nested dir is over"

    assert main(
        ["restructure", "--source", str(source), "--output", str(output), "--yes"]
    ) == 0

    # The nested directory keeps its position in the tree and gains numbered subfolders.
    nested_out = output / "day_two"
    assert nested_out.is_dir()
    assert sorted(p.name for p in nested_out.rglob("*.h5")) == [
        "ep_nested_a.h5",
        "ep_nested_b.h5",
    ]
    assert [p for p in nested_out.iterdir() if p.is_dir()], "the oversized dir is batched"

    # The root was within the limit, so it is copied through rather than reorganised.
    assert (output / "ep_root.h5").is_file(), "an in-limit root must not be restructured"

    # Every episode in the output still resolves its own video.
    for h5_path in output.rglob("*.h5"):
        with h5py.File(h5_path, "r") as f:
            stored = f["observations/video_paths/front"][()]
        rel = stored.decode() if isinstance(stored, bytes) else stored
        assert (h5_path.parent / rel).is_file(), f"{h5_path} lost its video"

    # Nothing is left behind: the root episode is in the output exactly once.
    assert len(list(output.rglob("ep_root.h5"))) == 1


def test_restructure_copies_untouched_directories_through(tmp_path, monkeypatch):
    """The output is a full copy, not just the episodes that needed splitting."""
    monkeypatch.setattr(restructure, "FILE_LIMIT", 2)
    source = tmp_path / "session"
    big = source / "big"
    big.mkdir(parents=True)
    write_valid_episode(big, "ep_a")
    write_valid_episode(big, "ep_b")  # 4 files > limit of 2
    small = source / "small"
    small.mkdir()
    write_valid_episode(small, "ep_small")
    (source / "NOTES.md").write_text("keep me", encoding="utf-8")
    output = tmp_path / "out"

    assert main(
        ["restructure", "--source", str(source), "--output", str(output), "--yes"]
    ) == 0

    assert (output / "NOTES.md").read_text(encoding="utf-8") == "keep me"
    assert (output / "small" / "ep_small.h5").is_file(), "small dirs pass through as-is"
    assert (output / "small" / "ep_small_front.mp4").is_file()
    assert not list((output / "small").glob("0*")), "an in-limit dir must not be batched"
    assert list((output / "big").glob("*/ep_a.h5")), "the oversized dir is batched"


def test_restructure_refuses_an_output_inside_the_source(tmp_path, monkeypatch, caplog):
    """Otherwise the pass-through copy would walk into its own output."""
    monkeypatch.setattr(restructure, "FILE_LIMIT", 1)
    source = tmp_path / "session"
    source.mkdir()
    write_valid_episode(source, "ep_a")

    assert main(
        ["restructure", "--source", str(source), "--output", str(source / "out"), "--yes"]
    ) == 1
    assert "must not be inside the source" in caplog.text


def test_restructure_reports_an_oversized_dir_it_cannot_split(tmp_path, monkeypatch, caplog):
    """A videos-only folder cannot be batched by episode; say so rather than claiming success."""
    monkeypatch.setattr(restructure, "FILE_LIMIT", 1)
    source = tmp_path / "session"
    source.mkdir()
    for i in range(3):
        (source / f"clip{i}.mp4").write_bytes(b"x")

    assert main(["restructure", "--source", str(source), "--yes"]) == 1
    assert "nothing to split" in caplog.text


def test_restructure_rejects_a_source_that_is_not_a_directory(tmp_path, caplog):
    episode = write_valid_episode(tmp_path, "ep_a")
    assert main(["restructure", "--source", str(episode), "--yes"]) == 1
    assert "not a directory" in caplog.text.lower()


def test_restructure_without_yes_aborts_on_anything_but_yes(tmp_path, monkeypatch, info_log):
    monkeypatch.setattr(restructure, "FILE_LIMIT", 0)
    write_valid_episode(tmp_path, "ep_a")
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    assert main(["restructure", "--source", str(tmp_path)]) == 0

    assert "Aborted" in info_log.text
    assert not (tmp_path.parent / f"{tmp_path.name}_restructured").exists()


# ── submissions ───────────────────────────────────────────────────────────────


class _FakeApi:
    def __init__(self, files=None):
        self.files = files
        self.token = None

    def repo_info(self, repo_id, repo_type):
        if self.files is None:
            raise RuntimeError("404")
        return {"id": repo_id}

    def list_repo_files(self, repo_id, repo_type):
        return self.files


@pytest.fixture
def fake_hf(monkeypatch):
    """Patch HfApi at its import site inside query_submissions."""

    def install(files):
        api = _FakeApi(files)
        monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: api)
        return api

    return install


def test_submissions_counts_episodes_videos_and_folders(fake_hf, info_log):
    fake_hf(["000/a.h5", "000/a_front.mp4", "500/b.hdf5", "README.md"])

    assert main(["submissions"]) == 0

    assert "Episodes (.h5): 2" in info_log.text
    assert "Videos (.mp4):  1" in info_log.text
    assert "Total files:    4" in info_log.text
    assert "000" in info_log.text and "500" in info_log.text


def test_submissions_treats_a_missing_repo_as_success(fake_hf, info_log):
    """The repo is created on first upload, so 'not there yet' is not a failure."""
    fake_hf(None)

    assert main(["submissions"]) == 0
    assert "created automatically on your first successful upload" in info_log.text


def test_submissions_lab_id_flag_overrides_the_config(fake_hf, info_log):
    fake_hf([])

    assert main(["submissions", "--lab-id", "SomeOtherLab"]) == 0
    assert "OopsieData-Submissions/SomeOtherLab" in info_log.text


def test_submissions_prefers_the_env_token(monkeypatch):
    """$HF_TOKEN overrides the config token, matching upload's precedence."""
    seen = {}

    def fake_api(token=None):
        seen["token"] = token
        return _FakeApi([])

    monkeypatch.setenv("HF_TOKEN", "env-token")
    monkeypatch.setattr("huggingface_hub.HfApi", fake_api)

    assert hf_upload.query_submissions() == 0
    assert seen["token"] == "env-token"
