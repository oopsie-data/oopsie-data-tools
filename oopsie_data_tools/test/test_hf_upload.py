"""Tests for the validate-and-upload helpers.

``utils/hf_upload.py`` is what stands between a lab and the public dataset, and it had no
tests at all. Nothing here touches the network: the HuggingFace API is a stand-in object,
and the only real work is on temporary directories.
"""

from __future__ import annotations

import logging

import pytest

from oopsie_data_tools.test.fixtures.make_valid import write_valid_episode
from oopsie_data_tools.utils import hf_upload
from oopsie_data_tools.utils.log import setup_logger


class FakeApi:
    """Records what an upload would have done, without doing it."""

    def __init__(self, existing_repos=()):
        self.existing = set(existing_repos)
        self.uploaded: list[tuple[str, str]] = []
        self.remote_files: list[str] = []

    def repo_info(self, repo_id, repo_type):
        if repo_id not in self.existing:
            raise RuntimeError("404")
        return {"id": repo_id}

    # No create_repo: the toolkit must never create a submissions repo, so an attempt to
    # call it here fails with AttributeError rather than passing silently.

    def upload_large_folder(self, folder_path, repo_id, repo_type):
        self.uploaded.append((repo_id, str(folder_path)))

    def list_repo_files(self, repo_id, repo_type):
        return self.remote_files


# ── run_validation ─────────────────────────────────────────────────────────────


def test_validation_passes_for_a_good_session(tmp_path):
    write_valid_episode(tmp_path, "ep_a")

    assert hf_upload.run_validation(str(tmp_path)) == 0


def test_validation_fails_for_a_bad_session(tmp_path):
    write_valid_episode(tmp_path, "ep_a")
    (tmp_path / "broken.h5").write_text("not hdf5", encoding="utf-8")

    assert hf_upload.run_validation(str(tmp_path)) == 1


def test_validation_reports_a_missing_path(tmp_path, caplog):
    assert hf_upload.run_validation(str(tmp_path / "nope")) == 1
    assert "does not exist" in caplog.text


def test_validation_of_a_single_episode_by_id(tmp_path):
    write_valid_episode(tmp_path, "000001")

    assert hf_upload.run_validation(str(tmp_path), episode_id="000001") == 0


def test_validation_writes_the_log_file_when_asked(tmp_path):
    write_valid_episode(tmp_path, "ep_a")
    log_path = tmp_path / "run.log"

    hf_upload.run_validation(str(tmp_path), log_path=str(log_path))

    assert log_path.exists() and log_path.stat().st_size > 0


# ── check_folder_size ──────────────────────────────────────────────────────────


def test_folder_size_check_passes_for_a_normal_layout(tmp_path):
    write_valid_episode(tmp_path, "ep_a")

    assert hf_upload.check_folder_size(str(tmp_path)) == []


def test_folder_size_check_flags_a_directory_over_the_limit(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(hf_upload, "FILE_LIMIT", 3)
    for i in range(5):
        (tmp_path / f"f{i}.bin").write_text("x", encoding="utf-8")

    oversized = hf_upload.check_folder_size(str(tmp_path))

    assert [count for _, count in oversized] == [5]
    assert "exceed" in caplog.text
    # The remedy must be a subcommand, not a path into the source checkout. It used to name
    # scripts/validate_and_upload/restructure_large_folder.py, which does not exist for anyone
    # who pip-installed the package — leaving the precheck's only fix unreachable.
    assert "oopsie-data restructure --source" in caplog.text, "tell the user how to fix it"
    assert "scripts/" not in caplog.text


def test_folder_size_result_is_falsy_when_fine(tmp_path):
    """cmd_upload branches on this directly, so its truthiness is load-bearing."""
    assert not hf_upload.check_folder_size(str(tmp_path))


# ── resolve_hf_target ──────────────────────────────────────────────────────────


def test_repo_is_derived_from_the_lab_id(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    token, repo = hf_upload.resolve_hf_target()

    assert repo == "OopsieData-Submissions/test_lab"  # conftest's stubbed contributor config
    assert token == "test_token"


def test_hf_token_env_var_overrides_the_config(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_environment")

    token, _ = hf_upload.resolve_hf_target()

    assert token == "hf_from_the_environment"


# ── require_repo / upload_dataset ─────────────────────────────────────────────


def test_missing_repo_aborts_instead_of_being_created(caplog):
    api = FakeApi()

    with caplog.at_level(logging.ERROR):
        assert hf_upload.require_repo(api, "OopsieData-Submissions/lab") is False

    assert "not found" in caplog.text
    assert "lab_id" in caplog.text


def test_existing_repo_passes_the_check():
    api = FakeApi(existing_repos={"OopsieData-Submissions/lab"})

    assert hf_upload.require_repo(api, "OopsieData-Submissions/lab") is True


def test_upload_sends_the_whole_folder(tmp_path):
    write_valid_episode(tmp_path, "ep_a")
    api = FakeApi()
    api.remote_files = ["ep_a.h5"]

    hf_upload.upload_dataset(api, "OopsieData-Submissions/lab", str(tmp_path))

    assert api.uploaded == [("OopsieData-Submissions/lab", str(tmp_path))]


def test_upload_warns_when_fewer_episodes_arrive_than_were_sent(tmp_path, caplog):
    write_valid_episode(tmp_path, "ep_a")
    write_valid_episode(tmp_path, "ep_b")
    api = FakeApi()
    api.remote_files = ["ep_a.h5"]  # one of the two is missing remotely

    with caplog.at_level(logging.WARNING):
        hf_upload.upload_dataset(api, "OopsieData-Submissions/lab", str(tmp_path))

    assert "below local" in caplog.text


def test_upload_survives_a_repo_listing_failure(tmp_path, caplog):
    """The post-upload check is a courtesy; it must not fail an upload that worked."""
    write_valid_episode(tmp_path, "ep_a")
    api = FakeApi()

    def boom(**_):
        raise RuntimeError("listing unavailable")

    api.list_repo_files = boom

    hf_upload.upload_dataset(api, "OopsieData-Submissions/lab", str(tmp_path))

    assert api.uploaded, "the upload itself still happened"
    assert "Could not confirm" in caplog.text


# ── utils/log.py ───────────────────────────────────────────────────────────────


def test_setup_logger_writes_to_the_named_file(tmp_path):
    log_path = tmp_path / "a.log"

    logger = setup_logger("oopsie_data_tools.test.logging_probe", str(log_path))
    logger.info("hello")

    assert "hello" in log_path.read_text(encoding="utf-8")


def test_setup_logger_does_not_stack_handlers(tmp_path):
    """It is called once per validated file, so repeated calls must not accumulate."""
    name = "oopsie_data_tools.test.logging_probe_repeat"

    first = setup_logger(name, str(tmp_path / "a.log"))
    handler_count = len(first.handlers)
    for _ in range(3):
        setup_logger(name, str(tmp_path / "a.log"))

    assert len(first.handlers) == handler_count


def test_setup_logger_honours_a_second_different_file(tmp_path):
    """The guard used to be "any handler at all", so a second --log-path wrote nothing."""
    name = "oopsie_data_tools.test.logging_probe_repeat"
    first_file, second_file = tmp_path / "a.log", tmp_path / "b.log"

    setup_logger(name, str(first_file))
    logger = setup_logger(name, str(second_file))
    logger.info("second run")

    assert "second run" in second_file.read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _detach_probe_loggers():
    """Loggers are global; leave none behind holding a temp file open."""
    yield
    for name in (
        "oopsie_data_tools.test.logging_probe",
        "oopsie_data_tools.test.logging_probe_repeat",
    ):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
