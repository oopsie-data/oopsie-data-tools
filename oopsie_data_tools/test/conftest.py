"""Shared pytest fixtures for oopsie_data_tools tests.

Session-scoped fixtures generate HDF5 + MP4 files once per test session into a
pytest-managed temporary directory and are cleaned up automatically at the end.

  tmp_path_factory  (session-scoped) – shared fixtures used across many tests
  tmp_path          (function-scoped, built-in) – per-test isolation

File generation delegates to the canonical fixture generators so that the
validator tests always exercise the exact same files as other test modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oopsie_data_tools.test.fixtures.make_invalid import (
    _MAKERS_SIMPLE,
    _MAKERS_WITH_VIDEO,
    _write_video,
)
from oopsie_data_tools.test.fixtures.make_valid import (
    make_failure,
    make_legacy_v1,
    make_multi_camera,
    make_no_annotations,
    make_success,
    make_unannotated,
    write_valid_episode,  # re-exported for per-test use in test modules
)

__all__ = ["write_valid_episode"]


@pytest.fixture(autouse=True)
def _isolate_user_environment(tmp_path_factory, monkeypatch):
    """Keep every test out of the developer's real home, config and working directory.

    Three confirmed leaks this closes:

    * ``cli.main`` exports ``$OOPSIE_CONFIG_DIR``. ``monkeypatch.delenv`` records nothing
      when the variable is absent, so there was nothing to undo and the value outlived the
      test that set it, pointing later tests at a deleted directory.
    * ``test_robot_setup`` resolves profiles through the cwd and
      ``$OOPSIE_ROBOT_PROFILES_DIR``, so it failed if either was set, or if pytest ran from
      a directory that happened to contain ``robot_profiles/``.
    * ``init_wizard.advise_persisting_config_dir`` appends an export line to ``~/.zshrc``.
      Nothing stopped that from being the developer's real one.
    """
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    for var in ("OOPSIE_CONFIG_DIR", "OOPSIE_ROBOT_PROFILES_DIR"):
        # Set before deleting so monkeypatch always has a value to restore, whether or not
        # the variable existed and whatever the test under it does.
        monkeypatch.setenv(var, str(home / "unset"))
        monkeypatch.delenv(var)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


@pytest.fixture(scope="session", autouse=True)
def _test_contributor_config():
    """Supply a test lab_id everywhere the contributor config is read.

    Lets the suite run on a fresh checkout without a filled-in
    ``configs/contributor_config.yaml`` (whose blank lab_id would otherwise fail
    ``EpisodeRecorder`` construction), without touching the committed file.

    Every module that binds the name at import gets patched: ``hf_upload`` and
    ``init_wizard`` also do ``from ... import read_contributor_config``, so rebinding two
    hand-picked modules left them reading the developer's real config.
    """
    from _pytest.monkeypatch import MonkeyPatch

    import oopsie_data_tools.annotation_tool.episode_recorder as _recorder
    import oopsie_data_tools.init_wizard as _init_wizard
    import oopsie_data_tools.utils.contributor_config as _cc
    import oopsie_data_tools.utils.hf_upload as _hf_upload

    def _fake(config_path=None) -> tuple[str, str]:
        return ("test_lab", "test_token")

    patch = MonkeyPatch()
    for module in (_cc, _recorder, _hf_upload, _init_wizard):
        patch.setattr(module, "read_contributor_config", _fake)
    try:
        yield
    finally:
        patch.undo()


# ---------------------------------------------------------------------------
# Session-scoped fixtures: valid episodes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def valid_episode(tmp_path_factory) -> Path:
    """Fully valid oopsiedata_format_v1 episode, minimally annotated as a success.

    It carries an annotation despite ``make_unannotated``'s name: validation through the
    CLI always checks annotations, so a general-purpose "valid episode" must have one.
    """
    d = tmp_path_factory.mktemp("valid")
    make_unannotated(d)
    return d / "episode_unannotated.h5"


@pytest.fixture(scope="session")
def episode_without_annotations(tmp_path_factory) -> Path:
    """Structurally valid episode with no ``episode_annotations`` group at all."""
    d = tmp_path_factory.mktemp("no_annotations")
    make_no_annotations(d)
    return d / "episode_no_annotations.h5"


@pytest.fixture(scope="session")
def valid_success_episode(tmp_path_factory) -> Path:
    """Valid episode annotated as success."""
    d = tmp_path_factory.mktemp("valid_success")
    make_success(d)
    return d / "episode_success.h5"


@pytest.fixture(scope="session")
def valid_failure_episode(tmp_path_factory) -> Path:
    """Valid episode annotated as failure."""
    d = tmp_path_factory.mktemp("valid_failure")
    make_failure(d)
    return d / "episode_failure.h5"


@pytest.fixture(scope="session")
def legacy_v1_episode(tmp_path_factory) -> Path:
    """Valid episode still carrying the taxonomy v1 annotation schema."""
    d = tmp_path_factory.mktemp("legacy_v1")
    make_legacy_v1(d)
    return d / "episode_legacy_v1.h5"


@pytest.fixture(scope="session")
def valid_session_dir(tmp_path_factory) -> Path:
    """Session directory containing all four valid episode types."""
    d = tmp_path_factory.mktemp("valid_session")
    make_unannotated(d)
    make_success(d)
    make_failure(d)
    make_multi_camera(d)
    return d


# ---------------------------------------------------------------------------
# Session-scoped fixture: invalid episodes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def invalid_fixtures(tmp_path_factory) -> dict[str, Path]:
    """Dict mapping fixture stem → Path for every invalid HDF5 file.

    Generated by the same make_invalid module used for manual inspection,
    so tests and the generator stay in sync automatically.
    """
    d = tmp_path_factory.mktemp("invalid")

    for maker in _MAKERS_SIMPLE:
        maker(d)

    for maker in _MAKERS_WITH_VIDEO:
        maker(d, _write_video)

    return {p.stem: p for p in d.glob("invalid_*.h5")}
