"""``oopsie-data annotation download`` and ``upload``, against a fake HuggingFace API."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

from oopsie_data_tools.cli import main
from oopsie_data_tools.test.fixtures.make_valid import _write_annotation
from oopsie_data_tools.utils import annotation_batches

NAME = "Jiahui Karen Chén"
FOLDER = "jiahui_karen_chen"


def _episode(path: Path, annotator: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["episode_id"] = path.stem
        if annotator:
            _write_annotation(f, annotator=annotator, outcome="failure")


class FakeApi:
    """Serves ``remote`` (repo path -> writer) and records uploads."""

    def __init__(self, remote: dict, *args, **kwargs):
        self.remote = remote
        self.uploads: list[dict] = []

    def list_repo_tree(self, repo, path_in_repo, repo_type, recursive):
        return [SimpleNamespace(path=p, blob_id="x") for p in self.remote if p.startswith(path_in_repo + "/")]

    def snapshot_download(self, repo, repo_type, allow_patterns, local_dir):
        prefix = allow_patterns[0].rstrip("*")
        for p, write in self.remote.items():
            if p.startswith(prefix):
                write(Path(local_dir) / p)

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)
        return SimpleNamespace(pr_url="https://huggingface.co/datasets/x/discussions/1")


@pytest.fixture
def api(monkeypatch):
    remote = {
        f"{FOLDER}/batch_1/episodes/0001.h5": _episode,
        f"{FOLDER}/batch_1/episodes/0002.h5": _episode,
        f"{FOLDER}/batch_2/episodes/0001.h5": _episode,
        "someone_else/batch_1/episodes/0001.h5": _episode,
    }
    fake = FakeApi(remote)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: fake)
    monkeypatch.setattr(annotation_batches, "check_token", lambda token: "hf_user")
    return fake


def test_folder_name_matches_the_coordinator_slug():
    assert annotation_batches.folder_name(NAME) == FOLDER
    assert annotation_batches.folder_name("  O'Brien, Mary-Ann ") == "o_brien_mary_ann"


def test_download_fetches_only_the_annotators_folder(api, tmp_path, capsys):
    out = tmp_path / "share"

    assert main(["annotation", "download", "--name", NAME, "--token", "t", "--dir", str(out)]) == 0

    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.h5")) == [
        f"{FOLDER}/batch_1/episodes/0001.h5",
        f"{FOLDER}/batch_1/episodes/0002.h5",
        f"{FOLDER}/batch_2/episodes/0001.h5",
    ]
    assert json.loads((out / "annotator.json").read_text()) == {"name": NAME, "folder": FOLDER}
    assert f'--annotator-name "{NAME}"' in capsys.readouterr().out


def test_download_rejects_an_unknown_name(api, tmp_path):
    assert main(["annotation", "download", "--name", "Nobody", "--token", "t", "--dir", str(tmp_path)]) == 1
    assert not (tmp_path / "annotator.json").exists()


def test_upload_opens_a_pull_request_with_the_annotated_files(api, tmp_path):
    main(["annotation", "download", "--name", NAME, "--token", "t", "--dir", str(tmp_path)])
    share = tmp_path / FOLDER
    _episode(share / "batch_1/episodes/0002.h5", annotator=NAME)
    _episode(share / "batch_2/episodes/0001.h5", annotator="someone else")

    assert main(["annotation", "upload", "--token", "t", "--dir", str(tmp_path), "--yes"]) == 0

    (upload,) = api.uploads
    assert upload["repo_id"] == annotation_batches.RETURNS_REPO
    assert upload["path_in_repo"] == FOLDER
    assert upload["create_pr"] is True
    assert upload["allow_patterns"] == ["batch_1/episodes/0002.h5"]


def test_upload_refuses_when_nothing_is_annotated_under_the_name(api, tmp_path):
    main(["annotation", "download", "--name", NAME, "--token", "t", "--dir", str(tmp_path)])

    assert main(["annotation", "upload", "--token", "t", "--dir", str(tmp_path), "--yes"]) == 1
    assert api.uploads == []


def test_upload_needs_a_downloaded_share(api, tmp_path):
    assert main(["annotation", "upload", "--token", "t", "--dir", str(tmp_path), "--yes"]) == 1
