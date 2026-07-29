"""Path-handling guards on the annotation server.

The server hands out files from ``--samples-dir`` and writes annotations into HDF5 files
named by a query parameter. Both of those take a caller-supplied path, and every rejection
branch was untested: the traversal guard on ``/videos-path/<path>``, and all four failure
modes of ``_safe_h5_path_from_query``.
"""

from __future__ import annotations

import h5py
import pytest

from oopsie_data_tools.annotation_tool import annotator_server
from oopsie_data_tools.test.fixtures.make_valid import write_valid_episode


@pytest.fixture
def client(tmp_path):
    """A configured server over a samples dir holding one valid episode."""
    write_valid_episode(tmp_path, "ep")
    annotator_server.configure_runtime(
        samples_dir=tmp_path, annotator_name="tester", browse_only=True
    )
    annotator_server.app.config.update(TESTING=True)
    with annotator_server.app.test_client() as test_client:
        yield test_client


@pytest.fixture
def secret(tmp_path):
    """A file outside the samples directory that must never be reachable."""
    outside = tmp_path.parent / "outside_the_samples_dir.txt"
    outside.write_text("private", encoding="utf-8")
    return outside


# ── /videos-path/<path> ────────────────────────────────────────────────────────


def test_serves_a_video_inside_the_samples_dir(client, tmp_path):
    mp4 = next(tmp_path.glob("*.mp4"))

    response = client.get(f"/videos-path/{mp4.name}")
    try:
        assert response.status_code == 200
        assert response.mimetype == "video/mp4"
        assert response.data[:4] != b"", "the file should actually stream"
    finally:
        # send_file keeps the handle open until the response is closed, and the suite runs
        # with filterwarnings=error, so a leaked ResourceWarning fails the test.
        response.close()


@pytest.mark.parametrize(
    "attempt",
    [
        "../outside_the_samples_dir.txt",
        "../../etc/passwd",
        "subdir/../../outside_the_samples_dir.txt",
        "%2e%2e/outside_the_samples_dir.txt",
    ],
)
def test_traversal_out_of_the_samples_dir_is_refused(client, secret, attempt):
    response = client.get(f"/videos-path/{attempt}")

    assert response.status_code in (403, 404), (
        f"{attempt!r} returned {response.status_code}; it must not be served"
    )
    assert b"private" not in response.data


def test_missing_video_is_404(client):
    assert client.get("/videos-path/no_such_video.mp4").status_code == 404


# ── ?path= on the HDF5 endpoints ───────────────────────────────────────────────


def test_sample_reads_an_episode_in_the_samples_dir(client, tmp_path):
    h5_name = next(tmp_path.glob("*.h5")).name

    response = client.get(f"/api/h5/sample?path={h5_name}")

    assert response.status_code == 200
    assert response.get_json()["metadata"]["rel_path"] == h5_name


@pytest.mark.parametrize(
    "query,expected_status,reason",
    [
        ("", 400, "path query parameter is required"),
        ("?path=", 400, "empty path"),
        ("?path=../escape.h5", 403, "escapes samples directory"),
        ("?path=notes.txt", 400, "must refer to an .h5 file"),
        ("?path=absent.h5", 404, "not found"),
    ],
)
def test_sample_rejects_bad_paths(client, query, expected_status, reason):
    response = client.get(f"/api/h5/sample{query}")

    assert response.status_code == expected_status, reason
    assert "error" in response.get_json()


def test_annotation_write_refuses_a_path_outside_the_samples_dir(client, secret):
    response = client.post(
        "/api/h5/annotations?path=../outside_the_samples_dir.txt",
        json={"outcome": "success"},
    )

    assert response.status_code in (400, 403)
    assert secret.read_text(encoding="utf-8") == "private", "the file must be untouched"


def test_annotation_write_lands_in_the_episode(client, tmp_path):
    h5_path = next(tmp_path.glob("*.h5"))

    response = client.post(
        f"/api/h5/annotations?path={h5_path.name}",
        json={"outcome": "success", "additional_notes": "looks fine"},
    )

    assert response.status_code == 200
    with h5py.File(h5_path, "r") as f:
        group = f["episode_annotations"]["tester"]
        assert float(group.attrs["success"]) == 1.0
        assert group.attrs["additional_notes"] == "looks fine"


def test_instruction_write_refuses_a_path_outside_the_samples_dir(client, secret):
    response = client.post(
        "/api/h5/instruction?path=../outside_the_samples_dir.txt",
        json={"instruction": "overwritten"},
    )

    assert response.status_code in (400, 403)
    assert secret.read_text(encoding="utf-8") == "private"


def test_instruction_rejects_an_empty_value(client, tmp_path):
    h5_name = next(tmp_path.glob("*.h5")).name

    response = client.post(f"/api/h5/instruction?path={h5_name}", json={"instruction": "  "})

    assert response.status_code == 400
