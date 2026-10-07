"""Tests for QA mode: bundle unpacking and the QA-notes routes of the annotator server."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from oopsie_data_tools.annotation_tool import qa_manifest
from oopsie_data_tools.annotation_tool.annotator_server import app, configure_runtime
from oopsie_data_tools.test.fixtures.make_valid import write_valid_episode

BUNDLE_PATH = "episodes/Lab_A/ep.h5"


def _make_bundle_dir(root: Path) -> Path:
    bundle = root / "qa_bundle_01"
    ep_dir = bundle / "episodes" / "Lab_A"
    ep_dir.mkdir(parents=True)
    write_valid_episode(ep_dir, stem="ep")
    manifest = {
        "bundle_id": "qa_bundle_01",
        "reviewer": "",
        "episodes": [
            {
                "qa_id": "qa_bundle_01-001",
                "dataset_repo": "Org/Lab_A",
                "repo_path": "ep.h5",
                "bundle_path": BUNDLE_PATH,
                "qa_notes": "",
            }
        ],
    }
    (bundle / qa_manifest.MANIFEST_NAME).write_text(json.dumps(manifest))
    return bundle


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    return _make_bundle_dir(tmp_path)


@pytest.fixture
def client(bundle: Path):
    configure_runtime(
        samples_dir=bundle,
        annotator_name="reviewer",
        qa_manifest=bundle / qa_manifest.MANIFEST_NAME,
    )
    app.config.update(TESTING=True)
    return app.test_client()


def test_note_is_saved_to_manifest_and_h5_untouched(client, bundle: Path) -> None:
    h5 = bundle / BUNDLE_PATH
    before = _sha(h5)

    resp = client.post(f"/api/qa/notes?path={BUNDLE_PATH}", json={"notes": "camera flickers"})

    assert resp.status_code == 200, resp.data
    entry = qa_manifest.read_manifest(bundle / qa_manifest.MANIFEST_NAME)["episodes"][0]
    assert entry["qa_notes"] == "camera flickers"
    assert entry["qa_reviewer"] == "reviewer"
    assert entry["dataset_repo"] == "Org/Lab_A"
    assert _sha(h5) == before


def test_h5_write_routes_are_forbidden(client, bundle: Path) -> None:
    h5 = bundle / BUNDLE_PATH
    before = _sha(h5)

    ann = client.post(f"/api/h5/annotations?path={BUNDLE_PATH}", json={"outcome": "failure"})
    instr = client.post(f"/api/h5/instruction?path={BUNDLE_PATH}", json={"instruction": "x"})

    assert ann.status_code == 403
    assert instr.status_code == 403
    assert _sha(h5) == before


def test_list_tick_follows_notes(client) -> None:
    def tick() -> int:
        (item,) = client.get("/api/h5/list").get_json()
        return item["annotation_tick_level"]

    assert tick() == 0
    client.post(f"/api/qa/notes?path={BUNDLE_PATH}", json={"notes": "ok"})
    assert tick() == 2


def test_sample_includes_all_annotations_and_qa_entry(client) -> None:
    data = client.get(f"/api/h5/sample?path={BUNDLE_PATH}").get_json()

    assert data["qa"]["qa_id"] == "qa_bundle_01-001"
    assert [a["annotator"] for a in data["all_annotations"]] == ["test_annotator"]
    assert data["all_annotations"][0]["outcome"] == "success"


def test_note_for_unlisted_episode_is_rejected(client, bundle: Path) -> None:
    write_valid_episode(bundle / "episodes" / "Lab_A", stem="other")

    resp = client.post("/api/qa/notes?path=episodes/Lab_A/other.h5", json={"notes": "x"})

    assert resp.status_code == 404


def test_qa_routes_absent_outside_qa_mode(tmp_path: Path) -> None:
    configure_runtime(samples_dir=tmp_path, annotator_name="a", browse_only=True)
    app.config.update(TESTING=True)

    assert app.test_client().get("/api/qa/notes").status_code == 404


def test_unpack_bundle_roundtrip_and_resume(tmp_path: Path) -> None:
    src = _make_bundle_dir(tmp_path / "src")
    zip_path = tmp_path / "qa_bundle_01.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src.parent).as_posix())

    out = tmp_path / "out"
    bundle_dir = qa_manifest.unpack_bundle(zip_path, out)
    manifest_path = qa_manifest.find_manifest(bundle_dir)
    qa_manifest.set_note(manifest_path, BUNDLE_PATH, "keep me", "r")

    assert qa_manifest.unpack_bundle(zip_path, out) == bundle_dir
    assert qa_manifest.read_manifest(manifest_path)["episodes"][0]["qa_notes"] == "keep me"


def test_unpack_bundle_rejects_unsafe_members(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("b/qa_manifest.json", "{}")
        zf.writestr("../escape.txt", "x")

    with pytest.raises(qa_manifest.QABundleError):
        qa_manifest.unpack_bundle(zip_path, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()
